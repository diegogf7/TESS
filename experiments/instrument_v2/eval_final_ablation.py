# All this code is from Claude
"""FINAL test-set evaluation for the instrument ablation. Runs ONCE.

Refuses to run unless FINAL_EVAL=YES (the test split is inspected here and
nowhere else in this experiment). All checkpoint / probe / learning-rate
selection happened upstream on validation only, recorded in the manifests.

Produces frozen-probe and fine-tuned results for random / jepa / supcon /
hybrid, paired chip-stratified bootstrap differences for the six decisive
comparisons, per-chip recall, confusion matrices, and a verdict summary.

NOTE: results are EXPLORATORY -- this 1,600-star test split was already
inspected by earlier experiments (step-1/2 diagnostics, sector14 JEPA eval).

Run:  FINAL_EVAL=YES python -m src.instrument_v2.eval_final_ablation
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.instrument_v2.ablation_config import ARMS, SEEDS
from src.instrument_v2.diagnose_chip_common_signal import chip_index, chip_name
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.select_pretrain_checkpoints import build_model, encode, fit_probe
from src.instrument_v2.train_sector14_jepa import effective_rank
from src.instrument_v2.train_sector14_matched_finetune import FineTuneClassifier, make_head
from src.loss_function.gapblind_fix import build_gapblind_jepa

SECTOR = 14
S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ABL_DIR = os.environ.get("ABL_DIR", os.path.join("artifacts", "instrument_v2", "ablation", os.environ.get("RUN_ID", "dev")))
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOTSTRAP = 1000

FROZEN_PAIRS = (("jepa", "random"), ("hybrid", "supcon"))
FINETUNE_PAIRS = (("jepa", "random"), ("supcon", "random"),
                  ("hybrid", "random"), ("hybrid", "supcon"))


def require_final_gate():
    if os.environ.get("FINAL_EVAL") != "YES":
        raise RuntimeError("Refusing to touch the test split: set FINAL_EVAL=YES "
                           "only after all validation-based selection is complete.")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def metrics_16way(yte, pred):
    return {"bacc_16way": balanced_accuracy_score(yte, pred),
            "bacc_camera": balanced_accuracy_score(yte // 4, pred // 4),
            "bacc_ccd": balanced_accuracy_score(yte % 4, pred % 4),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "per_chip_recall": recall_score(yte, pred, labels=list(range(16)),
                                            average=None, zero_division=0).tolist()}


def stratified_indices(y, rng):
    """Bootstrap resample of test stars WITHIN each chip (chip-stratified)."""
    idx = []
    for chip in np.unique(y):
        rows = np.flatnonzero(y == chip)
        idx.append(rng.choice(rows, len(rows), replace=True))
    return np.concatenate(idx)


def paired_stratified_diff(y, preds_a, preds_b, n_boot=N_BOOTSTRAP, seed=0):
    """Mean-over-seeds paired diff of bacc on identical chip-stratified resamples."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = stratified_indices(y, rng)
        d = np.mean([balanced_accuracy_score(y[idx], pa[idx])
                     - balanced_accuracy_score(y[idx], pb[idx])
                     for pa, pb in zip(preds_a, preds_b)])
        diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(np.mean([balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb)
                           for pa, pb in zip(preds_a, preds_b)]))
    return {"diff_mean": point, "diff_lo": float(lo), "diff_hi": float(hi),
            "p_not_better": float(np.mean(diffs <= 0))}


def save_confusions(confusions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(confusions)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, (title, cm) in zip(axes, confusions.items()):
        ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1),
                  cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ticks = range(16)
        ax.set_xticks(ticks, [chip_name(i)[3:] for i in ticks], rotation=90, fontsize=5)
        ax.set_yticks(ticks, [chip_name(i)[3:] for i in ticks], fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step1_pca_reference():
    path = os.path.join(SPLIT_DIR, "results.csv")
    if not os.path.exists(path):
        return None
    r = pd.read_csv(path)
    r = r[(r["representation"] == "shared_sector_1024") & (r["condition"] == "real")]
    if len(r) == 0:
        return None
    best = r.loc[r["bacc_16way"].idxmax()]
    return {"label": "PCA shared-grid (SUPERVISED class-aggregated reference: "
                     "chip labels group the training stars; NOT label-free)",
            "K": int(best["K"]), "bacc_16way": float(best["bacc_16way"]),
            "bacc_camera": float(best["bacc_camera"])}


def main():
    require_final_gate()
    os.makedirs(ABL_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}")
    with open(os.path.join(ABL_DIR, "pretrain_selection.json")) as fh:
        pre_sel = json.load(fh)
    with open(os.path.join(ABL_DIR, "finetune_selection.json")) as fh:
        ft_sel = json.load(fh)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    tic = df["TIC"].astype(str)
    part = np.where(tic.isin(train_tics), "train",
                    np.where(tic.isin(val_tics), "val", "test"))
    chips = np.array([chip_index(c, d) for c, d in zip(df["camera"], df["ccd"])])
    X, M = grid_frame(df, "shared", t_range)
    te = part == "test"
    yte16 = chips[te]
    print(f"FINAL EVAL on {int(te.sum())} test stars (exploratory: split "
          "inspected by earlier experiments)")

    rows, confusions = [], {}
    frozen_preds = {}                            # arm -> [preds per seed]
    for arm in ARMS:
        frozen_preds[arm] = []
        for seed in SEEDS:
            entry = pre_sel["arms"][arm][str(seed)]
            model = build_model(arm, seed, entry["checkpoint"])
            Z = encode(model, X, M)
            clf = fit_probe(Z[part == "train"], chips[part == "train"],
                            entry["probe_C"], entry["probe_pca"] or None)
            pred = clf.predict(Z[te])
            frozen_preds[arm].append(pred)
            met = metrics_16way(yte16, pred)
            rows.append({"protocol": "frozen", "arm": arm, "target": "camccd",
                         "seed": seed, "epoch": entry["epoch"],
                         "latent_std": float(Z[te].std(axis=0).mean()),
                         "effective_rank": effective_rank(Z[te]), **met})
            print(f"frozen {arm:8s} s{seed}  bacc16 {met['bacc_16way']:.4f}  "
                  f"cam {met['bacc_camera']:.4f}  ccd {met['bacc_ccd']:.4f}", flush=True)
        cm = confusion_matrix(yte16, frozen_preds[arm][0], labels=range(16))
        confusions[f"frozen {arm} (s0)"] = cm

    ft_preds = {}                                # (arm, target) -> [preds per seed]
    for arm in ARMS:
        for target in ("camera", "camccd"):
            sel = ft_sel[arm][target]
            n_classes = 16 if target == "camccd" else 4
            ft_preds[(arm, target)] = []
            for seed in SEEDS:
                encoder = build_gapblind_jepa().target_encoder
                model = FineTuneClassifier(encoder, make_head(n_classes, seed, target)).to(DEVICE)
                model.load_state_dict(torch.load(sel["checkpoints"][str(seed)],
                                                 map_location=DEVICE))
                model.eval()
                preds = []
                with torch.no_grad():
                    for start in range(0, int(te.sum()), BATCH):
                        sl = np.flatnonzero(te)[start:start + BATCH]
                        logits = model(torch.tensor(X[sl]).to(DEVICE),
                                       torch.tensor(M[sl]).to(DEVICE))
                        preds.append(logits.argmax(dim=1).cpu().numpy())
                pred = np.concatenate(preds)
                ft_preds[(arm, target)].append(pred)
                if target == "camccd":
                    met = metrics_16way(yte16, pred)
                else:
                    met = {"bacc_camera": balanced_accuracy_score(yte16 // 4, pred),
                           "macro_f1": f1_score(yte16 // 4, pred, average="macro",
                                                zero_division=0)}
                rows.append({"protocol": "finetuned", "arm": arm, "target": target,
                             "seed": seed, "backbone_lr": sel["backbone_lr"], **met})
                headline = met.get("bacc_16way", met["bacc_camera"])
                print(f"ft {arm:8s} {target:7s} s{seed}  headline {headline:.4f}", flush=True)
            if target == "camccd":
                confusions[f"finetuned {arm} (s0)"] = confusion_matrix(
                    yte16, ft_preds[(arm, target)][0], labels=range(16))

    # ---------------- paired chip-stratified bootstrap differences -------------
    paired = []
    for a, b in FROZEN_PAIRS:
        d = paired_stratified_diff(yte16, frozen_preds[a], frozen_preds[b])
        paired.append({"comparison": f"frozen:{a}-minus-{b}", "target": "camccd", **d})
    for a, b in FINETUNE_PAIRS:
        for target in ("camera", "camccd"):
            y = yte16 if target == "camccd" else yte16 // 4
            d = paired_stratified_diff(y, ft_preds[(a, target)], ft_preds[(b, target)])
            paired.append({"comparison": f"finetuned:{a}-minus-{b}", "target": target, **d})
    for p in paired:
        print(f"PAIRED {p['comparison']:34s} {p['target']:7s} "
              f"{p['diff_mean']:+.4f} [{p['diff_lo']:+.4f},{p['diff_hi']:+.4f}] "
              f"p(<=0) {p['p_not_better']:.4f}")

    # ---------------- aggregate + verdict ----------------
    def agg(protocol, arm, target, metric):
        vals = [r[metric] for r in rows if r["protocol"] == protocol
                and r["arm"] == arm and r["target"] == target and metric in r]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    def sig(name):
        p = next(x for x in paired if x["comparison"] == name
                 and x["target"] == "camccd")
        return p["diff_mean"] > 0 and p["diff_lo"] > 0

    jepa_beats_random_frozen = sig("frozen:jepa-minus-random")
    jepa_beats_scratch_ft = sig("finetuned:jepa-minus-random")
    hybrid_beats_supcon_frozen = sig("frozen:hybrid-minus-supcon")
    hybrid_beats_supcon_ft = sig("finetuned:hybrid-minus-supcon")
    hybrid_beats_scratch_ft = sig("finetuned:hybrid-minus-random")

    verdict = []
    if jepa_beats_random_frozen or jepa_beats_scratch_ft:
        verdict.append("Pure JEPA works: it improves over random/scratch "
                       f"(frozen: {jepa_beats_random_frozen}, fine-tuned: {jepa_beats_scratch_ft}).")
    else:
        verdict.append("Pure JEPA does NOT improve over random (frozen) or scratch "
                       "(fine-tuned) on the 16-way chip task.")
    if hybrid_beats_supcon_frozen or hybrid_beats_supcon_ft:
        verdict.append("JEPA contributes to the hybrid: hybrid beats SupCon-only "
                       f"(frozen: {hybrid_beats_supcon_frozen}, fine-tuned: {hybrid_beats_supcon_ft}).")
    elif hybrid_beats_scratch_ft:
        verdict.append("Contrastive supervision worked (hybrid beats scratch), but "
                       "JEPA added no demonstrated benefit: hybrid does not beat SupCon-only.")
    else:
        verdict.append("Neither hybrid-over-SupCon nor hybrid-over-scratch improvements "
                       "are significant; no JEPA contribution demonstrated.")
    verdict.append("EXPLORATORY: the 1,600-star test split was already inspected in "
                   "previous experiments (diagnostics + sector14 JEPA eval).")

    ref = step1_pca_reference()
    pd.DataFrame(rows).to_csv(os.path.join(ABL_DIR, "final_results.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(ABL_DIR, "paired_differences.csv"), index=False)
    with open(os.path.join(ABL_DIR, "final_results.json"), "w") as fh:
        json.dump({"git_commit": git_commit(), "rows": rows, "paired": paired,
                   "pca_reference": ref, "verdict": verdict,
                   "hybrid_weight": pre_sel.get("hybrid_weight")}, fh, indent=2, default=float)
    save_confusions(confusions, os.path.join(ABL_DIR, "final_confusion_matrices.png"))

    lines = ["# Instrument ablation — final report", "",
             f"git commit: {git_commit()}  |  hybrid weight: {pre_sel.get('hybrid_weight')}", "",
             "## Verdict"] + [f"- {v}" for v in verdict] + ["", "## Results (mean ± std over seeds)", ""]
    lines.append("| protocol | arm | target | bacc16 | camera | ccd |")
    lines.append("|---|---|---|---|---|---|")
    for protocol in ("frozen", "finetuned"):
        for arm in ARMS:
            for target in (("camccd",) if protocol == "frozen" else ("camccd", "camera")):
                b16 = agg(protocol, arm, target, "bacc_16way")
                cam = agg(protocol, arm, target, "bacc_camera")
                ccd = agg(protocol, arm, target, "bacc_ccd")
                lines.append(f"| {protocol} | {arm} | {target} | "
                             f"{b16[0]:.4f}±{b16[1]:.4f} | {cam[0]:.4f}±{cam[1]:.4f} | "
                             f"{ccd[0]:.4f}±{ccd[1]:.4f} |")
    lines += ["", "## Paired chip-stratified bootstrap differences", ""]
    lines.append("| comparison | target | diff | 95% CI | p(<=0) |")
    lines.append("|---|---|---|---|---|")
    for p in paired:
        lines.append(f"| {p['comparison']} | {p['target']} | {p['diff_mean']:+.4f} | "
                     f"[{p['diff_lo']:+.4f}, {p['diff_hi']:+.4f}] | {p['p_not_better']:.4f} |")
    if ref:
        lines += ["", f"Reference: {ref['label']} — bacc16 {ref['bacc_16way']:.4f} "
                      f"(K={ref['K']}), camera {ref['bacc_camera']:.4f}."]
    with open(os.path.join(ABL_DIR, "final_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote final_results.csv/json, paired_differences.csv, "
          f"final_confusion_matrices.png, final_summary.md to {ABL_DIR}")


if __name__ == "__main__":
    main()
