"""Three-stage hyperparameter search whose first requirement is avoiding collapse.

Non-negotiable rules, enforced in code rather than by convention:
  - Only trials with a checkpoint that passed EVERY collapse check are eligible
    for ranking. There is no "least collapsed" fallback anywhere.
  - If no trial produces a valid checkpoint, the search reports failure and
    proposes an expanded space. It never returns a collapsed model.
  - The test split is never loaded here.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch

from .collapse import CollapseThresholds
from .config import Part2Config
from .evaluate import evaluate_representation, summarise_seeds, write_report
from .real_data import git_sha
from .train import (DEVICE, ValidationProbe, add_common_args, configs_from_args,
                    load_frozen_part1, load_sources, train_part1, train_part2)

# ----------------------------------------------------------------- the space
SPACE = {
    "phi": ("loguniform", 0.5, 25.0),
    "lam": ("loguniform_with_zero", 0.1, 50.0, 0.10),   # 10% of draws are exactly 0
    "mu": ("loguniform", 1.0, 100.0),
    "nu": ("loguniform_biased_high", 0.01, 10.0, 0.50),  # half the draws from [1, 10]
    "lr": ("loguniform", 1e-5, 1e-3),
    "weight_decay": ("loguniform", 1e-8, 1e-3),
    "tau": ("choice", [0.99, 0.995, 0.999, 0.9995]),
    "batch_size": ("choice", [128, 256, 512]),
    "mask_range": ("choice", [(0.20, 0.40), (0.30, 0.50), (0.40, 0.60)]),
}

# The configuration already shown to collapse. Carried through the search as a
# control so its failure is recorded rather than assumed.
CONTROL = {"phi": 1.0, "lam": 25.0, "mu": 1.0, "nu": 0.01,
           "lr": 1e-3, "weight_decay": 1e-6, "tau": 0.999,
           "batch_size": 256, "mask_range": (0.30, 0.50), "_name": "control_known_collapse"}


def _loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_config(rng, idx):
    out = {}
    for key, spec in SPACE.items():
        kind = spec[0]
        if kind == "loguniform":
            out[key] = _loguniform(rng, spec[1], spec[2])
        elif kind == "loguniform_with_zero":
            out[key] = 0.0 if rng.random() < spec[3] else _loguniform(rng, spec[1], spec[2])
        elif kind == "loguniform_biased_high":
            lo, hi, frac_high = spec[1], spec[2], spec[3]
            out[key] = (_loguniform(rng, 1.0, hi) if rng.random() < frac_high
                        else _loguniform(rng, lo, hi))
        elif kind == "choice":
            out[key] = spec[1][rng.randrange(len(spec[1]))]
    out["_name"] = f"trial{idx:04d}"
    return out


def apply_config(base: Part2Config, hp: dict, steps=None):
    cfg = Part2Config(**base.__dict__)
    for k, v in hp.items():
        if k.startswith("_"):
            continue
        if k == "mask_range":
            cfg.mask_ratio_min, cfg.mask_ratio_max = float(v[0]), float(v[1])
        elif k in cfg.__dict__:
            setattr(cfg, k, v)
    if steps:
        cfg.steps = steps
    return cfg


# ----------------------------------------------------------------- ranking
LEAK_FIELDS = ("camera_bacc", "ccd_bacc", "area_bacc")


def _leakage(split_report):
    vals = [split_report.get(f) for f in LEAK_FIELDS]
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def rank_trials(trials, acc_band=0.01):
    """Lexicographic ranking over ELIGIBLE trials only.

    1. maximise validation PhyTS balanced accuracy
    2. within `acc_band` of the best, prefer lower physics-systematics correlation
    3. then lower camera / CCD / area leakage
    4. then higher effective rank, then lower off-diagonal covariance
    5. then lower validation invariance loss

    Returns (ordered, notes). `ordered` is empty when nothing is eligible.
    """
    eligible = [t for t in trials if t.get("eligible")]
    notes = {"n_trials": len(trials), "n_eligible": len(eligible)}
    if not eligible:
        notes["reason"] = "no trial produced a checkpoint passing every collapse check"
        return [], notes

    def acc(t):
        v = t["metrics"].get("physics_bacc")
        return v if v is not None and v == v else float("nan")

    accs = [acc(t) for t in eligible]
    have_acc = any(a == a for a in accs)
    notes["physics_bacc_available"] = have_acc

    if have_acc:
        best_acc = max(a for a in accs if a == a)
        band = [t for t in eligible if acc(t) == acc(t) and acc(t) >= best_acc - acc_band]
        rest = [t for t in eligible if t not in band]
        notes["best_physics_bacc"] = best_acc
        notes["n_in_accuracy_band"] = len(band)
    else:
        # Criterion 1 is unavailable (no labels). Rank on 2..5 and say so loudly.
        notes["warning"] = ("PhyTS labels absent: criterion 1 (physics balanced "
                            "accuracy) could not be applied. Ranking used "
                            "criteria 2-5 only and is NOT the specified ranking.")
        band, rest = eligible, []

    def key(t):
        m = t["metrics"]
        return (m.get("mean_abs_corr_with_systematics", float("inf")),
                _leakage(m) if _leakage(m) == _leakage(m) else float("inf"),
                -m.get("effective_rank", 0.0),
                m.get("offdiag_cov_rms", float("inf")),
                m.get("val_inv", float("inf")))

    ordered = sorted(band, key=key) + sorted(rest, key=lambda t: -acc(t))
    return ordered, notes


def expanded_space_suggestion():
    """What to widen when every trial collapses. Never a relaxed threshold."""
    return {
        "note": "No valid configuration found. Thresholds are NOT lowered.",
        "widen": {
            "mu": "raise the ceiling to 1000 -- the variance hinge may be too weak",
            "nu": "raise the ceiling to 100 -- nu is what stops dimensional redundancy",
            "lam": "cap at 5 -- a large correlation weight buys decorrelation by shrinking",
            "lr": "extend down to 1e-6",
            "tau": "add 0.9999 -- a slower target reduces the collapse pressure",
            "architecture": "widen the latent or add a projector head before the losses",
        },
    }


# ----------------------------------------------------------------- stages
def evaluate_trial(name, ckpt, part1, sources, cfg, seed, device, probe_split="val"):
    """Metrics for one trial's saved (non-collapsed) checkpoint. Validation only."""
    from .models import S4Encoder
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    enc = S4Encoder(c["physics_dim"], c["n_tokens"], c["d_model"],
                    c["d_state"], c["n_layers"], 0.0)
    enc.load_state_dict(blob["ema_encoder"])          # primary representation
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    enc = enc.to(device)

    rep = evaluate_representation(name, enc, part1,
                                  {"train": sources["train"], probe_split: sources[probe_split]},
                                  device, cfg, seed)
    m = dict(rep["splits"][probe_split])
    m["val_inv"] = blob["val"]["val_inv"]
    m["effective_rank"] = blob["val"]["effective_rank"]
    m["offdiag_cov_rms"] = blob["val"]["offdiag_cov_rms"]
    m["mean_abs_corr_with_systematics"] = blob["val"]["val_mean_abs_corr"]
    return m, rep


def run_stage(stage, hp_list, sources, part1_ckpt, base_cfg, steps, seeds, out_root,
              thresholds, device=DEVICE, prune_after=2):
    os.makedirs(out_root, exist_ok=True)
    part1 = load_frozen_part1(part1_ckpt, device)
    probe = ValidationProbe(sources["val"], batch_size=256, device=device)
    trials = []

    for hp in hp_list:
        name = hp["_name"]
        per_seed, ckpts = [], []
        for seed in seeds:
            cfg = apply_config(base_cfg, hp, steps)
            run_dir = os.path.join(out_root, f"{name}_seed{seed}")
            ckpt, summary, _ = train_part2(
                sources, part1_ckpt, cfg, seed, run_dir, device,
                thresholds=thresholds, probe=probe, prune_after=prune_after)
            per_seed.append(summary)
            if ckpt:
                ckpts.append(ckpt)

        # Eligible only when EVERY seed produced a valid checkpoint.
        eligible = bool(ckpts) and len(ckpts) == len(seeds)
        metrics = {}
        if eligible:
            mlist = []
            for seed, ckpt in zip(seeds, ckpts):
                m, _ = evaluate_trial(name, ckpt, part1, sources,
                                      apply_config(base_cfg, hp, steps), seed, device)
                mlist.append(m)
            keys = set().union(*[set(m) for m in mlist])
            for k in keys:
                vals = [m[k] for m in mlist if isinstance(m.get(k), (int, float))
                        and m[k] == m[k]]
                if vals:
                    metrics[k] = float(np.mean(vals))
                    metrics[k + "_sd"] = float(np.std(vals))

        t = {"name": name, "stage": stage, "hp": {k: v for k, v in hp.items()
                                                 if not k.startswith("_")},
             "seeds": list(seeds), "checkpoints": ckpts,
             "status": [s["status"] for s in per_seed],
             "collapse_free": [bool(s["collapse_free_after_warmup"]) for s in per_seed],
             "meets_target_rank": [bool(s["meets_target_rank"]) for s in per_seed],
             "eligible": eligible, "metrics": metrics}
        trials.append(t)
        with open(os.path.join(out_root, "trials.jsonl"), "a") as fh:
            fh.write(json.dumps(t, default=str) + "\n")
        print(f"  [{stage}] {name}: status={t['status']} eligible={eligible}", flush=True)

    return trials


def search(sources, part1_ckpt, base_cfg, out_root, n_trials=100,
           stage1_steps=2500, stage2_steps=10000, stage3_steps=30000,
           stage2_keep=12, stage3_keep=3, seed=0, device=DEVICE,
           thresholds=None):
    thresholds = thresholds or CollapseThresholds()
    os.makedirs(out_root, exist_ok=True)
    rng = random.Random(seed)

    hp_list = [dict(CONTROL)] + [sample_config(rng, i) for i in range(n_trials)]
    with open(os.path.join(out_root, "space.json"), "w") as fh:
        json.dump({"space": {k: str(v) for k, v in SPACE.items()},
                   "control": {k: str(v) for k, v in CONTROL.items()},
                   "thresholds": thresholds.to_dict(), "n_trials": len(hp_list),
                   "git_sha": git_sha()}, fh, indent=2)

    print(f"\n=== stage 1: {len(hp_list)} trials x {stage1_steps} steps ===", flush=True)
    s1 = run_stage("stage1", hp_list, sources, part1_ckpt, base_cfg, stage1_steps,
                   [seed], os.path.join(out_root, "stage1"), thresholds, device,
                   prune_after=2)
    ordered1, notes1 = rank_trials(s1)
    if not ordered1:
        return _no_valid_configuration(out_root, notes1, s1)

    keep1 = [t["hp"] | {"_name": t["name"]} for t in ordered1[:stage2_keep]]
    print(f"\n=== stage 2: {len(keep1)} configs x {stage2_steps} steps x 2 seeds ===",
          flush=True)
    s2 = run_stage("stage2", keep1, sources, part1_ckpt, base_cfg, stage2_steps,
                   [seed, seed + 1], os.path.join(out_root, "stage2"), thresholds,
                   device, prune_after=2)
    ordered2, notes2 = rank_trials(s2)
    if not ordered2:
        return _no_valid_configuration(out_root, notes2, s2)

    keep2 = [t["hp"] | {"_name": t["name"]} for t in ordered2[:stage3_keep]]
    print(f"\n=== stage 3: {len(keep2)} configs x {stage3_steps} steps x 3 seeds ===",
          flush=True)
    s3 = run_stage("stage3", keep2, sources, part1_ckpt, base_cfg, stage3_steps,
                   [0, 1, 2], os.path.join(out_root, "stage3"), thresholds, device,
                   prune_after=None)

    # A final configuration must stay non-collapsed THROUGHOUT training, on every seed.
    for t in s3:
        t["eligible"] = bool(t["eligible"]) and all(t["collapse_free"])
    ordered3, notes3 = rank_trials(s3)
    if not ordered3:
        notes3["reason"] = ("every stage-3 configuration collapsed at some point after "
                            "warmup; none is demonstrably collapse-free throughout")
        return _no_valid_configuration(out_root, notes3, s3)

    winner = ordered3[0]
    target_met = all(winner["meets_target_rank"])
    result = {
        "selected": winner["name"],
        "hp": winner["hp"],
        "metrics": winner["metrics"],
        "collapse_free_all_seeds": all(winner["collapse_free"]),
        "effective_rank": winner["metrics"].get("effective_rank"),
        "target_effective_rank": thresholds.target_effective_rank,
        "meets_target_rank": target_met,
        "ranking_notes": {"stage1": notes1, "stage2": notes2, "stage3": notes3},
        "selected_on": "validation only -- the test split was never loaded",
        "git_sha": git_sha(),
    }
    if not target_met:
        result["target_rank_warning"] = (
            f"The selected model does not reach the target effective rank of "
            f"{thresholds.target_effective_rank:g}/64. The target was NOT lowered. "
            f"Treat this as an unmet objective and expand the search.")
        result["expanded_space"] = expanded_space_suggestion()

    with open(os.path.join(out_root, "selection.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print_selection(result)
    return result


def _no_valid_configuration(out_root, notes, trials):
    out = {"selected": None,
           "status": "NO VALID CONFIGURATION FOUND",
           "notes": notes,
           "n_trials": len(trials),
           "expanded_space": expanded_space_suggestion(),
           "git_sha": git_sha()}
    with open(os.path.join(out_root, "selection.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\n" + "=" * 70)
    print("NO VALID CONFIGURATION FOUND -- every trial collapsed.")
    print(f"  {notes.get('reason', '')}")
    print("  Thresholds were NOT lowered and no collapsed model was selected.")
    print("  Suggested expansion:")
    for k, v in out["expanded_space"]["widen"].items():
        print(f"    {k}: {v}")
    print("=" * 70)
    return out


def print_selection(r):
    print("\n" + "=" * 70)
    print(f"SELECTED: {r['selected']}")
    for k, v in r["hp"].items():
        print(f"  {k:>14} = {v}")
    m = r["metrics"]
    print(f"  effective rank {m.get('effective_rank', float('nan')):.1f} / 64 "
          f"(target {r['target_effective_rank']:g}) -> "
          f"{'MET' if r['meets_target_rank'] else 'NOT MET'}")
    print(f"  physics bacc {m.get('physics_bacc', float('nan')):.4f}  "
          f"corr {m.get('mean_abs_corr_with_systematics', float('nan')):.4f}  "
          f"val_inv {m.get('val_inv', float('nan')):.4f}")
    print(f"  collapse-free on every seed: {r['collapse_free_all_seeds']}")
    if not r["meets_target_rank"]:
        print("\n  " + r["target_rank_warning"])
    print("=" * 70)


def selected_config_path(out_root):
    return os.path.join(out_root, "selection.json")


def load_selected(out_root, base_cfg):
    """Requirement 5: the chosen hyperparameters flow into later stages."""
    with open(selected_config_path(out_root)) as fh:
        sel = json.load(fh)
    if not sel.get("selected"):
        raise RuntimeError("no valid configuration was selected; refusing to "
                           "propagate a collapsed or absent configuration")
    return apply_config(base_cfg, sel["hp"]), sel


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--part1-ckpt", required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--stage1-steps", type=int, default=2500)
    ap.add_argument("--stage2-steps", type=int, default=10000)
    ap.add_argument("--stage3-steps", type=int, default=30000)
    ap.add_argument("--stage2-keep", type=int, default=12)
    ap.add_argument("--stage3-keep", type=int, default=3)
    args = ap.parse_args()

    sources = load_sources(args)
    if "test" in sources:
        del sources["test"]            # the search must not be able to touch it
    _, c2 = configs_from_args(args)
    search(sources, args.part1_ckpt, c2, args.out, args.n_trials,
           args.stage1_steps, args.stage2_steps, args.stage3_steps,
           args.stage2_keep, args.stage3_keep, args.seed)
