"""Pilot sweep (spec s5) and the control / ablation matrix (spec s6).

The sweep selects on VALIDATION ONLY. Test numbers are produced once, by
final_eval.py, after a configuration has been chosen.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from .config import Part1Config, Part2Config
from .evaluate import evaluate_representation, summarise_seeds, write_report, is_collapsed
from .models import S4Encoder, VICRegJEPA
from .train import (DEVICE, load_sources, add_common_args, configs_from_args,
                    train_part1, train_part2, load_frozen_part1)

# spec s5
PILOT_GRID = {"lam": [5.0, 25.0], "mu": [1.0, 25.0]}

# spec s6 -- each entry mutates a Part2Config and/or picks a different encoder
ABLATIONS = {
    "random_init":   {"kind": "random_encoder"},
    "no_disent":     {"kind": "jepa", "lam": 0.0},
    "no_varcov":     {"kind": "jepa", "mu": 0.0, "nu": 0.0},
    "full_vicreg":   {"kind": "jepa"},
    "random_part1":  {"kind": "jepa", "random_init_part1": True},
}


def apply_overrides(cfg, overrides):
    cfg = Part2Config(**cfg.__dict__)
    for k, v in overrides.items():
        if k in cfg.__dict__:
            setattr(cfg, k, v)
    return cfg


def run_one(name, sources, part1_ckpt, base_cfg, seed, out_root, overrides=None,
            device=DEVICE):
    """Train (or instantiate) one representation and evaluate it on validation."""
    overrides = dict(overrides or {})
    kind = overrides.pop("kind", "jepa")
    random_part1 = overrides.pop("random_init_part1", False)
    cfg = apply_overrides(base_cfg, overrides)
    run_dir = os.path.join(out_root, f"{name}_seed{seed}")

    part1 = load_frozen_part1(part1_ckpt, device)

    if kind == "random_encoder":
        torch.manual_seed(seed)
        enc = S4Encoder(cfg.physics_dim, cfg.n_tokens, cfg.d_model,
                        cfg.d_state, cfg.n_layers, 0.0).to(device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        os.makedirs(run_dir, exist_ok=True)
        summary = {"note": "untrained control"}
        eval_enc = enc
    else:
        _, summary, model = train_part2(sources, part1_ckpt, cfg, seed, run_dir,
                                        device, random_init_part1=random_part1)
        # spec s7: the primary representation is the EMA encoder on the full curve
        eval_enc = model.ema_encoder

    report = evaluate_representation(name, eval_enc, part1,
                                     {k: sources[k] for k in ("train", "val")},
                                     device, cfg, seed)
    report["train_summary"] = summary
    report["overrides"] = {**overrides, "kind": kind, "random_init_part1": random_part1}
    write_report(report, os.path.join(run_dir, "eval.json"))
    return report


def pilot(sources, part1_ckpt, base_cfg, seed, out_root, device=DEVICE):
    """spec s5: short real-data pilots over lambda x mu; reject collapsed configs."""
    rows = []
    for lam, mu in itertools.product(PILOT_GRID["lam"], PILOT_GRID["mu"]):
        name = f"lam{lam:g}_mu{mu:g}"
        print(f"\n=== pilot {name} ===", flush=True)
        rep = run_one(name, sources, part1_ckpt, base_cfg, seed, out_root,
                      {"lam": lam, "mu": mu}, device)
        v = rep["splits"]["val"]
        rows.append({"config": name, "lam": lam, "mu": mu,
                     "val_inv": rep["train_summary"].get("best_val_inv", float("nan")),
                     "mean_abs_corr": v["mean_abs_corr_with_systematics"],
                     "std_min": v["std_min"], "std_median": v["std_median"],
                     "frac_std_below_0.5": v["frac_std_below_0.5"],
                     "offdiag_cov_rms": v["offdiag_cov_rms"],
                     "effective_rank": v["effective_rank"],
                     "collapsed": v["collapsed"]})

    survivors = [r for r in rows if not r["collapsed"]]
    chosen = None
    if survivors:
        # among non-collapsed, prefer the lowest physics-systematics correlation,
        # breaking ties on effective rank. Validation only -- test is untouched.
        chosen = sorted(survivors, key=lambda r: (r["mean_abs_corr"], -r["effective_rank"]))[0]
    out = {"grid": rows, "survivors": [r["config"] for r in survivors],
           "selected": chosen["config"] if chosen else None,
           "selection_rule": "min mean|corr| among non-collapsed, tie-break max effective rank",
           "selected_on": "validation only"}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "pilot.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print_table(rows)
    print(f"\nselected: {out['selected']}  (from {len(survivors)}/{len(rows)} non-collapsed)")
    return out


def print_table(rows):
    cols = ["config", "val_inv", "mean_abs_corr", "std_min", "std_median",
            "frac_std_below_0.5", "effective_rank", "collapsed"]
    w = {c: max(len(c), 12) for c in cols}
    print("  " + "  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(w[c]))
        print("  " + "  ".join(cells))


def ablations(sources, part1_ckpt, base_cfg, seeds, out_root, device=DEVICE,
              only=None):
    """spec s6: controls and ablations, each over several seeds."""
    results = {}
    for name, ov in ABLATIONS.items():
        if only and name not in only:
            continue
        reps = []
        for seed in seeds:
            print(f"\n=== ablation {name} seed {seed} ===", flush=True)
            reps.append(run_one(name, sources, part1_ckpt, base_cfg, seed,
                                out_root, dict(ov), device))
        results[name] = {"seeds": seeds, "per_seed": reps,
                         "summary": summarise_seeds(reps, "val")}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "ablations.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\n=== ablation summary (validation, mean +/- sd over seeds) ===")
    for name, r in results.items():
        s = r["summary"]
        def fmt(k):
            return f"{s[k]['mean']:.4f}+/-{s[k]['std']:.4f}" if k in s else "n/a"
        print(f"  {name:<14} physics={fmt('physics_bacc')}  "
              f"corr={fmt('mean_abs_corr_with_systematics')}  "
              f"erank={fmt('effective_rank')}")
    return results


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--stage", choices=["pilot", "ablations"], default="pilot")
    ap.add_argument("--part1-ckpt", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    sources = load_sources(args)
    _, c2 = configs_from_args(args)
    if args.stage == "pilot":
        pilot(sources, args.part1_ckpt, c2, args.seed, args.out)
    else:
        ablations(sources, args.part1_ckpt, c2, args.seeds, args.out, only=args.only)
