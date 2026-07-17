# All this code is from Claude
"""FINAL online-vs-EMA encoder audit report. Requires FINAL_EVAL=YES.

Assembles: (A) same-checkpoint frozen online-minus-EMA (from the fixed audit's
saved predictions), (B) independently selected frozen comparisons (online
selection is new; the EMA-selected side IS abl1's selection, reproduced in the
fixed audit), and (C) fine-tuned comparisons (online fine-tunes vs abl1's EMA
fine-tunes and scratch). Paired chip-stratified bootstrap CIs throughout.

Not publication-grade: this test split was inspected by earlier experiments.

Run:  FINAL_EVAL=YES python -m src.instrument_v2.eval_encoder_source_final
Out:  $NEW_RUN/encoder_source_results.{csv,json},
      $NEW_RUN/encoder_source_differences.csv, $NEW_RUN/final_summary.md,
      $NEW_RUN/encoder_source_confusions.png
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix

from src.instrument_v2.ablation_config import ONLINE_FT_ARMS, SEEDS
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.encoder_source import encode_features
from src.instrument_v2.eval_final_ablation import (
    metrics_16way, paired_stratified_diff, save_confusions,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.select_pretrain_checkpoints import build_model, fit_probe
from src.instrument_v2.train_sector14_jepa import effective_rank
from src.instrument_v2.train_sector14_matched_finetune import FineTuneClassifier, make_head
from src.loss_function.gapblind_fix import build_gapblind_jepa

SECTOR = 14
S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
OLD_RUN = os.environ.get("OLD_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1"))
NEW_RUN = os.environ.get("NEW_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1_encoder_audit"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256
SCRATCH_REFERENCE = 0.5144                       # abl1 fine-tuned scratch camccd


def require_final_gate():
    if os.environ.get("FINAL_EVAL") != "YES":
        raise RuntimeError("Refusing final test evaluation: set FINAL_EVAL=YES only "
                           "after all validation-based selection is complete.")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def predict_finetuned(ckpt_path, X, M, te):
    model = FineTuneClassifier(build_gapblind_jepa().target_encoder,
                               make_head(16, 0, "camccd")).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    preds = []
    idx = np.flatnonzero(te)
    with torch.no_grad():
        for start in range(0, len(idx), BATCH):
            sl = idx[start:start + BATCH]
            logits = model(torch.tensor(X[sl]).to(DEVICE), torch.tensor(M[sl]).to(DEVICE))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def sig(diff):
    return diff["diff_mean"] > 0 and diff["diff_lo"] > 0


def main():
    require_final_gate()
    os.makedirs(NEW_RUN, exist_ok=True)
    print(f"git commit: {git_commit()}")
    fixed = np.load(os.path.join(NEW_RUN, "fixed_preds.npz"))
    with open(os.path.join(NEW_RUN, "fixed_results.json")) as fh:
        fixed_rows = json.load(fh)["rows"]
    with open(os.path.join(NEW_RUN, "online_pretrain_selection.json")) as fh:
        online_sel = json.load(fh)
    with open(os.path.join(NEW_RUN, "online_finetune_selection.json")) as fh:
        online_ft = json.load(fh)
    with open(os.path.join(OLD_RUN, "finetune_selection.json")) as fh:
        old_ft = json.load(fh)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    tic = df["TIC"].astype(str)
    part = np.where(tic.isin(train_tics), "train",
                    np.where(tic.isin(val_tics), "val", "test"))
    chips = np.array([chip_index(c, d) for c, d in zip(df["camera"], df["ccd"])])
    X, M = grid_frame(df, "shared", t_range)
    tr, te = part == "train", part == "test"
    y = chips[te]
    assert np.array_equal(y, fixed["y_test"]), "test labels drifted vs fixed audit"

    rows, confusions = list(fixed_rows), {}
    for r in rows:
        r.setdefault("protocol", "frozen_fixed")

    # ---- B: frozen, independently selected ONLINE encoders ----
    sel_preds = {}
    for arm in ONLINE_FT_ARMS:
        sel_preds[arm] = []
        for seed in SEEDS:
            entry = online_sel["arms"][arm][str(seed)]
            model = build_model(arm, seed, entry["checkpoint"])
            Z = encode_features(model, "online", X, M, DEVICE)
            clf = fit_probe(Z[tr], chips[tr], entry["probe_C"], entry["probe_pca"] or None)
            pred = clf.predict(Z[te])
            sel_preds[arm].append(pred)
            met = metrics_16way(y, pred)
            rows.append({"protocol": "frozen_online_selected", "arm": arm, "seed": seed,
                         "source": "online", "epoch": entry["epoch"],
                         "latent_std": float(Z[te].std(axis=0).mean()),
                         "effective_rank": effective_rank(Z[te]), **met})
            print(f"frozen-online-sel {arm:8s} s{seed} ep{entry['epoch']:3d}  "
                  f"bacc16 {met['bacc_16way']:.4f}", flush=True)
        confusions[f"online-sel {arm} (s0)"] = confusion_matrix(y, sel_preds[arm][0],
                                                                labels=range(16))

    def fixed_pred_list(arm, source):
        return [fixed[f"{arm}_s{s}_{source}"] for s in SEEDS]

    # ---- C: fine-tuned predictions ----
    ft_preds = {}
    for arm in ONLINE_FT_ARMS:
        ft_preds[("online", arm)] = [predict_finetuned(online_ft[arm]["checkpoints"][str(s)],
                                                       X, M, te) for s in SEEDS]
        ft_preds[("ema", arm)] = [predict_finetuned(old_ft[arm]["camccd"]["checkpoints"][str(s)],
                                                    X, M, te) for s in SEEDS]
    ft_preds[("scratch", "random")] = [predict_finetuned(
        old_ft["random"]["camccd"]["checkpoints"][str(s)], X, M, te) for s in SEEDS]
    for (kind, arm), plist in ft_preds.items():
        for seed, pred in zip(SEEDS, plist):
            met = metrics_16way(y, pred)
            rows.append({"protocol": f"finetuned_{kind}", "arm": arm, "seed": seed,
                         "source": kind, **met})
            print(f"ft-{kind:7s} {arm:8s} s{seed}  bacc16 {met['bacc_16way']:.4f}", flush=True)

    # ---- paired chip-stratified differences ----
    diffs = []

    def add(name, a, b):
        d = paired_stratified_diff(y, a, b)
        diffs.append({"comparison": name, **d})
        print(f"PAIRED {name:44s} {d['diff_mean']:+.4f} "
              f"[{d['diff_lo']:+.4f},{d['diff_hi']:+.4f}] p(<=0) {d['p_not_better']:.4f}")

    for arm in ONLINE_FT_ARMS:                    # A: same checkpoint
        add(f"A_fixed:online-minus-ema:{arm}",
            fixed_pred_list(arm, "online"), fixed_pred_list(arm, "ema"))
    random_preds = fixed_pred_list("random", "identical")
    add("B_selected:online_jepa-minus-random", sel_preds["jepa"], random_preds)
    add("B_selected:online_supcon-minus-random", sel_preds["supcon"], random_preds)
    add("B_selected:online_hybrid-minus-online_supcon", sel_preds["hybrid"], sel_preds["supcon"])
    for arm in ONLINE_FT_ARMS:                    # B: online-selected vs abl1 EMA-selected
        add(f"B_selected:online-minus-ema:{arm}", sel_preds[arm], fixed_pred_list(arm, "ema"))
    for arm in ONLINE_FT_ARMS:                    # C: fine-tuned
        add(f"C_ft:online-minus-ema:{arm}", ft_preds[("online", arm)], ft_preds[("ema", arm)])
        add(f"C_ft:online_{arm}-minus-scratch", ft_preds[("online", arm)],
            ft_preds[("scratch", "random")])
    add("C_ft:online_hybrid-minus-online_supcon",
        ft_preds[("online", "hybrid")], ft_preds[("online", "supcon")])

    # ---- verdict ----
    by_name = {d["comparison"]: d for d in diffs}
    online_beats_ema = any(sig(by_name[f"A_fixed:online-minus-ema:{a}"])
                           or sig(by_name[f"B_selected:online-minus-ema:{a}"])
                           or sig(by_name[f"C_ft:online-minus-ema:{a}"])
                           for a in ONLINE_FT_ARMS)
    online_beats_scratch = any(sig(by_name[f"C_ft:online_{a}-minus-scratch"])
                               for a in ONLINE_FT_ARMS)
    if online_beats_ema and online_beats_scratch:
        verdict = "Online encoder rescues the representation."
    elif online_beats_ema:
        verdict = ("Online encoder improves performance but does not beat "
                   "scratch/SupCon.")
    else:
        verdict = ("Online and EMA are effectively equivalent; the original JEPA "
                   "conclusion stands.")
    notes = [verdict,
             f"Scratch reference (abl1 fine-tuned random, camccd): {SCRATCH_REFERENCE:.4f}.",
             "NOT publication-grade: this 1,600-star test split was inspected by "
             "previous experiments (diagnostics, sector14 JEPA eval, abl1)."]

    pd.DataFrame(rows).to_csv(os.path.join(NEW_RUN, "encoder_source_results.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(NEW_RUN, "encoder_source_differences.csv"), index=False)
    with open(os.path.join(NEW_RUN, "encoder_source_results.json"), "w") as fh:
        json.dump({"git_commit": git_commit(), "rows": rows, "diffs": diffs,
                   "verdict": notes,
                   "online_hybrid_weight": online_sel.get("hybrid_weight")},
                  fh, indent=2, default=float)
    save_confusions(confusions, os.path.join(NEW_RUN, "encoder_source_confusions.png"))

    def agg(protocol, arm, source=None):
        vals = [r["bacc_16way"] for r in rows if r["protocol"] == protocol
                and r["arm"] == arm and (source is None or r.get("source") == source)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),) * 2

    lines = ["# Encoder-source audit (online vs EMA) — final report", "",
             f"git commit: {git_commit()}  |  online hybrid weight: "
             f"{online_sel.get('hybrid_weight')}", "", "## Verdict"]
    lines += [f"- {n}" for n in notes]
    lines += ["", "## Frozen results (bacc16, mean ± std over seeds)", "",
              "| protocol | arm | source | bacc16 |", "|---|---|---|---|"]
    for arm in ("random",) + ONLINE_FT_ARMS:
        for proto, src in (("frozen_fixed", "online"), ("frozen_fixed", "ema"),
                           ("frozen_fixed", "identical"), ("frozen_online_selected", "online")):
            m, s = agg(proto, arm, src)
            if not np.isnan(m):
                lines.append(f"| {proto} | {arm} | {src} | {m:.4f}±{s:.4f} |")
    lines += ["", "## Fine-tuned results (bacc16, camccd)", "",
              "| init | arm | bacc16 |", "|---|---|---|"]
    for (kind, arm) in ft_preds:
        m, s = agg(f"finetuned_{kind}", arm)
        lines.append(f"| {kind} | {arm} | {m:.4f}±{s:.4f} |")
    lines += ["", "## Paired chip-stratified bootstrap differences", "",
              "| comparison | diff | 95% CI | p(<=0) |", "|---|---|---|---|"]
    for d in diffs:
        lines.append(f"| {d['comparison']} | {d['diff_mean']:+.4f} | "
                     f"[{d['diff_lo']:+.4f}, {d['diff_hi']:+.4f}] | {d['p_not_better']:.4f} |")
    with open(os.path.join(NEW_RUN, "final_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote encoder_source_results.csv/json, encoder_source_differences.csv, "
          f"confusions, final_summary.md to {NEW_RUN}")


if __name__ == "__main__":
    main()
