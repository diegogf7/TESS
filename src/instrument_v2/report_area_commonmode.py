# All this code is from Claude
"""Selection + reporting for the area common-mode experiment (val-only).

  STAGE=screen_select  pick best area and best chip config from the seed-0
                       screen selections, enforcing the anti-collapse gates
                       (erank >= 16, same-area cos >= 0.90, same > cross).
                       Refuses to promote (exit 1) if an arm has no gated
                       survivor -- the pipeline chain then stops.

  STAGE=final          aggregate frozen + LP-FT results over seeds, compute
                       paired chip-stratified bootstrap CIs on the validation
                       stars, and write final_summary.{md,json}.

Test TICs are never loaded in either stage.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.area_commonmode_jepa import build_area_commonmode_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_area_commonmode_finetune import Classifier, make_head
from src.instrument_v2.train_group_level_matched_finetune import build_frames
from src.instrument_v2.train_sector14_jepa import git_commit

STAGE = os.environ.get("STAGE", "final")
ART_DIR = os.environ.get(
    "ACM_ART_DIR", os.path.join("artifacts", "instrument_v2", "area_commonmode_v1"))
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "group_level"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
SEEDS = (0, 1, 2)
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "2000"))
SCRATCH_REF = 0.5559     # matched scratch S4D validation bacc16 (group-level run)
SUCCESS_MEAN = 0.565
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


# ------------------------------------------------------------ screen_select
def screen_select():
    selections = [load_json(p) for p in
                  sorted(glob.glob(os.path.join(ART_DIR, "selection_acm_*_s0.json")))]
    if not selections:
        raise SystemExit("no seed-0 screen selections found")
    chosen, table = {}, []
    for grouping in ("area", "chip"):
        candidates = [s for s in selections
                      if not s.get("skipped") and s["grouping"] == grouping]
        table.extend(candidates)
        gated = [s for s in candidates if s["passes_gates"]]
        if not gated:
            print(f"screen_select: NO {grouping} configuration passed the "
                  f"anti-collapse gates -- refusing to promote", flush=True)
            chosen[grouping] = None
            continue
        # primary: frozen val probe; secondary (tie / near-tie): target stability
        gated.sort(key=lambda s: (round(s["best_val_probe_bacc16"], 4),
                                  s["best_same_group_cos"]), reverse=True)
        chosen[grouping] = {"grouping": grouping,
                            "target": gated[0]["target"],
                            "k": gated[0]["k"],
                            "screen": gated[0]}
    out = {"chosen": chosen, "git_commit": git_commit(),
           "screen_table": [{k: s.get(k) for k in
                             ("tag", "grouping", "target", "k", "skipped",
                              "best_val_probe_bacc16", "best_effective_rank",
                              "best_same_group_cos", "best_cross_group_cos",
                              "passes_gates")} for s in selections]}
    path = os.path.join(ART_DIR, "screen_selection.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out["chosen"], indent=2, default=str), flush=True)
    if chosen["area"] is None or chosen["chip"] is None:
        raise SystemExit(1)
    print(f"screen selection -> {path}", flush=True)


# ------------------------------------------------------------------- final
def best_ft_result(arm, seed):
    """Validation-only backbone-LR selection: max val bacc over LR results."""
    paths = glob.glob(os.path.join(ART_DIR, f"result_lpft_{arm}_s{seed}_lr*.json"))
    if not paths:
        raise SystemExit(f"missing LP-FT results for {arm} seed {seed}")
    results = [load_json(p) for p in paths]
    return max(results, key=lambda r: r["best_val_bacc16"])


def val_predictions(result, flux, mask, is_train):
    encoder = build_area_commonmode_jepa().context_encoder
    model = Classifier(encoder, make_head()).to(DEVICE)
    model.load_state_dict(torch.load(result["checkpoint"], map_location=DEVICE))
    model.eval()
    rows = np.flatnonzero(~is_train)
    predicted = np.empty(len(rows), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            index = rows[start:start + 256]
            logits = model(torch.from_numpy(flux[index]).to(DEVICE),
                           torch.from_numpy(mask[index]).to(DEVICE))
            predicted[start:start + 256] = logits.argmax(dim=1).cpu().numpy()
    return predicted


def paired_bootstrap(y_true, chips, pred_a, pred_b, n_boot=N_BOOTSTRAP, seed=0):
    """Chip-stratified paired bootstrap of bacc(a) - bacc(b) on val stars."""
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(chips == chip) for chip in np.unique(chips)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows[rng.integers(0, len(rows), len(rows))]
                              for rows in strata])
        diffs[b] = (balanced_accuracy_score(y_true[idx], pred_a[idx])
                    - balanced_accuracy_score(y_true[idx], pred_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def final_report():
    screen = load_json(os.path.join(ART_DIR, "screen_selection.json"))
    chosen = screen["chosen"]
    configs = {"area_cm": chosen["area"], "chip_cm": chosen["chip"]}

    # frozen probes over seeds (confirmation selections + group-JEPA baseline)
    frozen = {}
    for arm, config in configs.items():
        tag = f"acm_{config['grouping']}_{config['target']}_k{config['k']}"
        frozen[arm] = {seed: load_json(os.path.join(
            ART_DIR, f"selection_{tag}_s{seed}.json"))["best_val_probe_bacc16"]
            for seed in SEEDS}
    frozen["groupjepa"] = {}
    for seed in SEEDS:
        path = os.path.join(GROUP_ART_DIR,
                            f"selection_s14groupmean_k8_s{seed}.json")
        frozen["groupjepa"][seed] = (load_json(path)["best_val_probe_bacc16"]
                                     if os.path.exists(path) else float("nan"))

    # fine-tuned (validation-only LR selection per arm x seed)
    finetuned = {arm: {seed: best_ft_result(arm, seed) for seed in SEEDS}
                 for arm in ("scratch", "groupjepa", "chip_cm", "area_cm")}
    ft_scores = {arm: {seed: finetuned[arm][seed]["best_val_bacc16"]
                       for seed in SEEDS} for arm in finetuned}

    # paired chip-stratified bootstrap on validation predictions
    frame = pd.read_parquet(S14_DATA)
    frame = frame[frame["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, frame, train_tics)
    flux, mask, labels, is_train = build_frames(
        frame, train_tics, val_tics, test_tics, time_range)
    val_labels = labels[~is_train]

    predictions = {arm: {seed: val_predictions(finetuned[arm][seed], flux,
                                               mask, is_train)
                         for seed in SEEDS} for arm in finetuned}
    bootstrap = {}
    for other in ("chip_cm", "groupjepa", "scratch"):
        per_seed = [paired_bootstrap(val_labels, val_labels,  # strata = chip = label
                                     predictions["area_cm"][seed],
                                     predictions[other][seed], seed=seed)
                    for seed in SEEDS]
        bootstrap[f"area_minus_{other}"] = {
            "per_seed": per_seed,
            "mean_diff": float(np.mean([d[0] for d in per_seed]))}

    area_mean = float(np.mean(list(ft_scores["area_cm"].values())))
    verdict = {
        "beats_chip_cm": all(ft_scores["area_cm"][s] > ft_scores["chip_cm"][s]
                             for s in SEEDS),
        "beats_groupjepa": all(ft_scores["area_cm"][s] > ft_scores["groupjepa"][s]
                               for s in SEEDS),
        "beats_scratch": all(ft_scores["area_cm"][s] > ft_scores["scratch"][s]
                             for s in SEEDS),
        "area_mean_val_bacc16": area_mean,
        "mean_target_0565": area_mean >= SUCCESS_MEAN,
        "scratch_reference_group_run": SCRATCH_REF,
    }
    verdict["success"] = (verdict["beats_chip_cm"] and verdict["beats_groupjepa"]
                          and verdict["beats_scratch"])

    summary = {"git_commit": git_commit(), "chosen_configs": configs,
               "screen_table": screen["screen_table"],
               "frozen_val_bacc16": frozen, "finetuned_val_bacc16": ft_scores,
               "bootstrap_area_diffs": bootstrap, "verdict": verdict,
               "test_untouched": ("Test TICs were never loaded, selected on, "
                                  "or evaluated in this experiment; all "
                                  "numbers are validation-only.")}
    json_path = os.path.join(ART_DIR, "final_summary.json")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# Area common-mode JEPA -- final summary (validation-only)", "",
             f"git commit: {summary['git_commit']}", "",
             f"chosen area config: {configs['area_cm']['target']} "
             f"K={configs['area_cm']['k']}",
             f"chosen chip config: {configs['chip_cm']['target']} "
             f"K={configs['chip_cm']['k']}", "",
             "## Frozen val bacc16 (seeds 0/1/2)"]
    for arm, scores in frozen.items():
        lines.append(f"- {arm}: " + " / ".join(f"{scores[s]:.4f}" for s in SEEDS))
    lines += ["", "## LP-FT val bacc16 (seeds 0/1/2, val-selected backbone LR)"]
    for arm in ("scratch", "groupjepa", "chip_cm", "area_cm"):
        scores = ft_scores[arm]
        lines.append(f"- {arm}: " + " / ".join(f"{scores[s]:.4f}" for s in SEEDS)
                     + f"  (mean {np.mean(list(scores.values())):.4f})")
    lines += ["", "## Paired chip-stratified bootstrap (area minus X, val stars)"]
    for name, entry in bootstrap.items():
        per_seed = ", ".join(f"s{s}: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
                             for s, d in zip(SEEDS, entry["per_seed"]))
        lines.append(f"- {name}: mean {entry['mean_diff']:+.4f}  ({per_seed})")
    lines += ["", "## Verdict", "```", json.dumps(verdict, indent=2), "```", "",
              summary["test_untouched"], ""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"final report -> {md_path}", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    if STAGE == "screen_select":
        screen_select()
    elif STAGE == "final":
        final_report()
    else:
        raise SystemExit(f"unknown STAGE {STAGE!r}")
