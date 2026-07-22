# All this code is from Claude
"""Selection + reporting for area_commonmode_v2 (decorrelation experiment).

Unlike v1, EVERY exit path is rc 0 and STAGE=final ALWAYS writes
final_summary.{md,json} -- refusals are recorded, never silently cancelled.

  STAGE=screen_select  choose the covariance weight from the seed-0 screen
                       (gates: erank >= 16, same-area cos >= 0.90 > cross,
                       frozen bacc16 > 0.44). Writes screen_selection.json
                       with either the winner or the refusal reason.

  STAGE=final          aggregate whatever exists: raw diagnostic, screen
                       sweep, confirmation seeds, LP-FT arms, paired
                       chip-stratified bootstrap, effective-rank
                       trajectories. Missing stages are reported as skipped.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.area_commonmode_jepa import build_area_commonmode_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_area_commonmode_finetune import (
    Classifier,
    build_frames,
    make_head,
)
from src.instrument_v2.train_sector14_jepa import git_commit

STAGE = os.environ.get("STAGE", "final")
ART_DIR = os.environ.get(
    "ACM2_ART_DIR", os.path.join("artifacts", "instrument_v2", "area_commonmode_v2"))
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
FT_ARMS = ("scratch", "groupjepa", "v1_area", "v2_area")
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "2000"))
SCRATCH_REF = 0.5559
V1_REFERENCE = {"probe_bacc16": 0.4199, "note":
                "v1 area median_mad K=8 seed-0 screen (gates refused; "
                "all v1 probes 0.383-0.420, random-init reference 0.444)"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def maybe(path):
    return load_json(path) if os.path.exists(path) else None


# ------------------------------------------------------------ screen_select
def screen_select():
    os.makedirs(ART_DIR, exist_ok=True)
    diagnostic = maybe(os.path.join(ART_DIR, "raw_region_diagnostic.json"))
    selections = [load_json(p) for p in sorted(glob.glob(
        os.path.join(ART_DIR, "selection_acm_area_median_mad_k8_cw*_s0.json")))]

    out = {"git_commit": git_commit(), "selected": None, "refusal_reason": None,
           "diagnostic_passed": bool(diagnostic and diagnostic.get("passes")),
           "screen_table": [{k: s.get(k) for k in
                             ("tag", "cov_weight", "skipped",
                              "best_val_probe_bacc16", "best_effective_rank",
                              "best_same_group_cos", "best_cross_group_cos",
                              "gates", "passes_gates")} for s in selections]}
    if diagnostic is None:
        out["refusal_reason"] = "raw region diagnostic never ran"
    elif not diagnostic.get("passes"):
        out["refusal_reason"] = ("raw region diagnostic FAILED: no raw "
                                 "area-specific common-mode signal on val")
    elif not selections:
        out["refusal_reason"] = "no screen selections found"
    else:
        gated = [s for s in selections
                 if not s.get("skipped") and s.get("passes_gates")]
        if not gated:
            out["refusal_reason"] = ("no covariance weight passed the gates "
                                     "(erank>=16, same_cos>=0.90>cross, "
                                     "probe>0.44)")
        else:
            gated.sort(key=lambda s: (round(s["best_val_probe_bacc16"], 4),
                                      s["best_same_group_cos"]), reverse=True)
            out["selected"] = {"cov_weight": gated[0]["cov_weight"],
                               "screen": gated[0]}

    path = os.path.join(ART_DIR, "screen_selection.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps({"selected": out["selected"] is not None,
                      "refusal_reason": out["refusal_reason"]}, indent=2),
          flush=True)
    print(f"screen selection -> {path} (exit 0 either way)", flush=True)


# ------------------------------------------------------------------- final
def erank_trajectories():
    out = {}
    for path in sorted(glob.glob(os.path.join(ART_DIR, "metrics_acm_*.csv"))):
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
        out[os.path.basename(path)] = [round(float(r["effective_rank"]), 2)
                                       for r in rows if r.get("effective_rank")]
    return out


def best_ft_result(arm, seed):
    paths = glob.glob(os.path.join(ART_DIR, f"result_lpft_{arm}_s{seed}_lr*.json"))
    if not paths:
        return None
    return max((load_json(p) for p in paths),
               key=lambda r: r["best_val_bacc16"])


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


def paired_bootstrap(y_true, pred_a, pred_b, seed=0):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    diffs = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = np.concatenate([rows[rng.integers(0, len(rows), len(rows))]
                              for rows in strata])
        diffs[b] = (balanced_accuracy_score(y_true[idx], pred_a[idx])
                    - balanced_accuracy_score(y_true[idx], pred_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def final_report():
    os.makedirs(ART_DIR, exist_ok=True)
    diagnostic = maybe(os.path.join(ART_DIR, "raw_region_diagnostic.json"))
    screen = maybe(os.path.join(ART_DIR, "screen_selection.json"))
    selected = screen.get("selected") if screen else None

    summary = {"git_commit": git_commit(),
               "raw_region_diagnostic": diagnostic,
               "screen": screen,
               "v1_reference": V1_REFERENCE,
               "scratch_reference_group_run": SCRATCH_REF,
               "effective_rank_trajectories": erank_trajectories(),
               "confirmation": None, "finetune": None, "bootstrap": None,
               "test_untouched": ("Test TICs were never loaded, selected on, "
                                  "or evaluated anywhere in area_commonmode_v2; "
                                  "all numbers are validation-only.")}

    if selected is not None:
        cov_weight = selected["cov_weight"]
        frozen = {}
        for seed in SEEDS:
            sel = maybe(os.path.join(
                ART_DIR,
                f"selection_acm_area_median_mad_k8_cw{cov_weight:g}_s{seed}.json"))
            frozen[seed] = sel["best_val_probe_bacc16"] if sel else None
        summary["confirmation"] = {"cov_weight": cov_weight,
                                   "frozen_val_bacc16": frozen}

        results = {arm: {seed: best_ft_result(arm, seed) for seed in SEEDS}
                   for arm in FT_ARMS}
        if all(results[arm][seed] for arm in FT_ARMS for seed in SEEDS):
            ft_scores = {arm: {seed: results[arm][seed]["best_val_bacc16"]
                               for seed in SEEDS} for arm in FT_ARMS}
            summary["finetune"] = ft_scores

            frame = pd.read_parquet(S14_DATA)
            frame = (frame[frame["sector"] == 14].drop_duplicates("TIC")
                     .reset_index(drop=True))
            train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR,
                                                            BASE_ART_DIR)
            time_range = ensure_time_range(BASE_ART_DIR, frame, train_tics)
            flux, mask, labels, is_train = build_frames(
                frame, train_tics, val_tics, test_tics, time_range)
            val_labels = labels[~is_train]
            preds = {arm: {seed: val_predictions(results[arm][seed], flux,
                                                 mask, is_train)
                           for seed in SEEDS} for arm in FT_ARMS}
            bootstrap = {}
            for other in ("v1_area", "groupjepa", "scratch"):
                per_seed = [paired_bootstrap(val_labels,
                                             preds["v2_area"][seed],
                                             preds[other][seed], seed=seed)
                            for seed in SEEDS]
                bootstrap[f"v2_minus_{other}"] = {
                    "per_seed": per_seed,
                    "mean_diff": float(np.mean([d[0] for d in per_seed]))}
            summary["bootstrap"] = bootstrap
        else:
            summary["finetune"] = "incomplete -- some LP-FT results missing"

    json_path = os.path.join(ART_DIR, "final_summary.json")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# area_commonmode_v2 -- final summary (validation-only)", "",
             f"git commit: {summary['git_commit']}", ""]
    if diagnostic:
        verdict = "PASS" if diagnostic.get("passes") else "FAIL"
        val = diagnostic["splits"]["val"]["same_minus_cross"]["combined_r"]
        lines += ["## Raw region diagnostic: " + verdict,
                  f"val same-minus-cross combined r: {val['mean']:+.4f} "
                  f"[{val['ci95'][0]:+.4f}, {val['ci95'][1]:+.4f}]", ""]
    else:
        lines += ["## Raw region diagnostic: NEVER RAN", ""]
    if screen:
        lines += ["## Screen (seed 0, area median_mad K=8)"]
        for row in screen.get("screen_table", []):
            lines.append(f"- cw={row.get('cov_weight')}: "
                         f"probe {row.get('best_val_probe_bacc16')}, "
                         f"erank {row.get('best_effective_rank')}, "
                         f"gates {row.get('passes_gates')}")
        lines += ["", ("### Selected: cov_weight "
                       f"{selected['cov_weight']}" if selected else
                       f"### REFUSED: {screen.get('refusal_reason')}"), ""]
    if summary.get("confirmation"):
        lines += ["## Confirmation (frozen val bacc16 per seed)",
                  json.dumps(summary["confirmation"], indent=2), ""]
    if isinstance(summary.get("finetune"), dict):
        lines += ["## LP-FT val bacc16 (val-selected backbone LR)"]
        for arm in FT_ARMS:
            scores = summary["finetune"][arm]
            lines.append(f"- {arm}: "
                         + " / ".join(f"{scores[s]:.4f}" for s in SEEDS)
                         + f"  (mean {np.mean(list(scores.values())):.4f})")
        lines += ["", "## Paired bootstrap (v2 minus X, val stars)"]
        for name, entry in summary["bootstrap"].items():
            per_seed = ", ".join(f"s{s}: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
                                 for s, d in zip(SEEDS, entry["per_seed"]))
            lines.append(f"- {name}: mean {entry['mean_diff']:+.4f} ({per_seed})")
        lines.append("")
    lines += ["## Effective-rank trajectories (per metrics CSV)",
              json.dumps(summary["effective_rank_trajectories"], indent=2), "",
              summary["test_untouched"], ""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"final report -> {md_path}", flush=True)


if __name__ == "__main__":
    if STAGE == "screen_select":
        screen_select()
    elif STAGE == "final":
        final_report()
    else:
        raise SystemExit(f"unknown STAGE {STAGE!r}")
