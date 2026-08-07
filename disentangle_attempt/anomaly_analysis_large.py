"""Scaled-up anomaly analysis: 20k chip-balanced anchors and 90%-variance PCA.

Isolates two changes against the earlier 1000-anchor / fixed-32-and-16-dim run:
  1. 20x more anchors, sampled evenly over the sector/camera/CCD chips rather than
     linearly through a concatenated list, which under-weights small chips;
  2. PCA sized by retained variance (>= 0.90) instead of a fixed dimension -- 16
     components held only ~0.51 of the instrument latent across 80 chips.

The flow architecture and training procedure are unchanged, so any difference is
attributable to those two changes. The disentanglement model is frozen and asserted
unchanged; only scalers, PCAs and flows are fitted, on the training split alone.

A global flow could in principle learn CHIP IDENTITY rather than unusual behaviour
within a chip, so per-chip candidate concentration is measured and warned about.

    python -m disentangle_attempt.anomaly_analysis_large \
      --checkpoint .../multichip_5sectors_v1/best.pt --parquet ... \
      --output-dir .../multichip_5sectors_v1/anomaly_analysis_20k_pca90 \
      --old-scores .../snapshot_20260806_134803/anomaly_flows/anomaly_scores.csv
"""

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from disentangle_attempt.dataset import (CrossSectorPatch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.fit_anomaly_flows import THRESHOLD, fit_flow, nll, percentile_against
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import build_model
from disentangle_attempt.plot_latent_umaps import instrument_latents, physics_latents
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

PER_CHIP = {"train": 200, "val": 25, "test": 25}
MIN_VARIANCE = 0.90
BATCH_SIZE = 256
UMAP_MAX = 10000
UMAP_KWARGS = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=42)


# ----------------------------------------------------------------------- sampling
def safe_spearman(a, b):
    """nan for a constant input (e.g. one chip, so sector never varies)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b)[0])


def sample_anchors(patch, seed):
    """Even per-chip draw, respecting each anchor's EXISTING split assignment."""
    rng = np.random.default_rng(seed)
    index_of = {split: {int(a): i for i, a in enumerate(patch.split_anchors[split])}
                for split in PER_CHIP}
    rows, splits, shortfalls = [], [], []
    for chip in patch.chips:
        for split, want in PER_CHIP.items():
            available = np.asarray(
                [int(a) for a in patch.split_anchors[split]
                 if patch.chip_of_row.get(int(a)) == chip], dtype=np.int64)
            take = min(want, len(available))
            if take < want:
                shortfalls.append({"sector": chip[0], "camera": chip[1], "ccd": chip[2],
                                   "split": split, "wanted": want, "available": int(len(available))})
            if take:
                picked = available[rng.permutation(len(available))[:take]]
                rows.extend(int(r) for r in np.sort(picked))
                splits.extend([split] * take)
    rows = np.asarray(rows, dtype=np.int64)
    assert len(set(rows)) == len(rows), "an anchor was sampled twice"
    peer_rows = np.stack([patch.peers[split][0][index_of[split][int(row)]]
                          for row, split in zip(rows, splits)])
    return rows, splits, peer_rows, shortfalls


def fit_pca(latents, is_train, seed, label, out_dir):
    """StandardScaler + smallest PCA reaching MIN_VARIANCE, fit on train only."""
    scaler = StandardScaler().fit(latents[is_train])
    scaled_train = scaler.transform(latents[is_train])
    full = PCA(random_state=seed).fit(scaled_train)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, MIN_VARIANCE) + 1)
    pca = PCA(n_components=n_components, random_state=seed).fit(scaled_train)
    retained = float(pca.explained_variance_ratio_.sum())
    assert retained >= MIN_VARIANCE - 1e-6, f"{label}: retained {retained:.4f} < {MIN_VARIANCE}"
    joblib.dump(scaler, os.path.join(out_dir, f"{label}_scaler.joblib"))
    joblib.dump(pca, os.path.join(out_dir, f"{label}_pca.joblib"))
    print(f"  {label}: 512 -> {n_components} components, retained variance {retained:.4f}",
          flush=True)
    return pca.transform(scaler.transform(latents)), n_components, retained, cumulative


# -------------------------------------------------------------------------- plots
def pca_curve_plot(curves, dims, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, cumulative in curves.items():
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, label=
                f"{label} (chose {dims[label]})")
        ax.axvline(dims[label], color="0.7", lw=0.7, ls=":")
    ax.axhline(MIN_VARIANCE, color="0.3", lw=0.8, ls="--", label=f"{MIN_VARIANCE:.0%} target")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def nll_plot(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, column, title in ((axes[0], "physics_nll", "physics flow"),
                              (axes[1], "instrument_nll", "instrument flow")):
        for split, colour in (("train", "0.6"), ("val", "tab:orange"), ("test", "tab:blue")):
            values = table.loc[table["split"] == split, column]
            if len(values):
                ax.hist(values, bins=60, histtype="step", color=colour, label=split,
                        density=True)
        ax.set_xlabel("negative log-likelihood")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def comparison_plot(table, path):
    colours = {"typical": "0.75", "physics": "tab:red", "instrument": "tab:blue",
               "both": "tab:purple"}
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name, colour in colours.items():
        pick = table["classification"] == name
        ax.scatter(table.loc[pick, "instrument_percentile"],
                   table.loc[pick, "physics_percentile"], s=3, color=colour,
                   label=f"{name} ({int(pick.sum())})", linewidths=0)
    ax.axvline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.axhline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("instrument anomaly percentile")
    ax.set_ylabel("physics anomaly percentile")
    ax.set_title("four interpretation regions", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", markerscale=3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def umap_plot(coords, percentile, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = np.argsort(percentile)
    art = ax.scatter(coords[order, 0], coords[order, 1], c=percentile[order], s=3,
                     cmap="viridis", vmin=0, vmax=1, linewidths=0)
    high = percentile >= THRESHOLD
    ax.scatter(coords[high, 0], coords[high, 1], s=18, facecolors="none",
               edgecolors="red", linewidths=0.5, label=f"percentile >= {THRESHOLD}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + "\n(UMAP is visualization only; distance is not the score)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.colorbar(art, ax=ax, fraction=0.04, label="anomaly percentile")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def chip_plot(per_chip, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(per_chip))
    ax.bar(x - 0.2, per_chip["physics_rate"], 0.4, color="tab:red", label="physics")
    ax.bar(x + 0.2, per_chip["instrument_rate"], 0.4, color="tab:blue", label="instrument")
    ax.axhline(0.05, color="0.3", lw=0.8, ls="--", label="0.05 expected by construction")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(per_chip["chip"][::2], rotation=90, fontsize=5)
    ax.set_ylabel("fraction of the chip's anchors flagged")
    ax.set_title("candidate rate per sector/camera/CCD -- a chip far above 0.05 suggests "
                 "the flow is reading chip identity, not within-chip behaviour", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def gallery(patch, entries, curves, title, path):
    if not entries:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"{title}: no candidates in this category", ha="center")
        ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(entries), 1, figsize=(12, 2.1 * len(entries)))
    axes = np.atleast_1d(axes)
    for ax, (row, label) in zip(axes, entries):
        valid = patch.M[row]
        x = np.arange(len(valid))
        show = lambda a: np.where(valid, a, np.nan)
        ax.plot(x, show(patch.X[row]), lw=0.7, color="0.6", label="raw")
        drawn = curves.get(row)
        if drawn:
            ax.plot(x, show(drawn["reconstructed"]), lw=0.7, color="tab:green",
                    label="reconstructed")
            ax.plot(x, show(drawn["cleaned"]), lw=0.7, color="tab:blue", label="cleaned")
            ax.plot(x, show(drawn["correction"]), lw=0.7, color="tab:red", label="correction")
            ax.plot(x, show(drawn["peer_common"]), lw=0.7, color="tab:olive",
                    label="peer common mode")
        ax.set_ylabel(label, fontsize=6)
    axes[0].legend(fontsize=6, ncol=5, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    note = "" if curves else "  (raw only: reference_context.pt not available)"
    fig.suptitle(title + note, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--old-scores", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    sector, camera, ccd = target_from_checkpoint(state, config)
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(config, args.require_cross_sector),
        verbose=False)
    print(f"{len(patch.chips)} chips, {len(patch.eligible_rows)} eligible anchors",
          flush=True)

    rows, splits, peer_rows, shortfalls = sample_anchors(patch, args.seed)
    splits = np.asarray(splits)
    counts = {s: int((splits == s).sum()) for s in PER_CHIP}
    print(f"sampled {len(rows)} anchors: {counts}", flush=True)
    if shortfalls:
        print(f"  {len(shortfalls)} chip/split combinations short of target", flush=True)

    # A TIC must not straddle splits (splits are TIC-keyed, but verify on the sample).
    by_tic = pd.DataFrame({"tic": patch.tic[rows], "split": splits})
    assert by_tic.groupby("tic")["split"].nunique().max() == 1, "a TIC spans splits"

    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    masks = complementary_masks(config["curve_length"], n_masks=4)

    print(f"extracting latents on {device}", flush=True)
    physics = physics_latents(model, patch, rows, masks, device)
    instrument = instrument_latents(model, patch, peer_rows, device)
    weights_unchanged = all(torch.equal(v.cpu(), state["model"][k].cpu())
                            for k, v in model.state_dict().items())
    assert weights_unchanged, "model weights changed during extraction"
    assert np.isfinite(physics).all() and np.isfinite(instrument).all()
    np.save(os.path.join(args.output_dir, "physics_latents.npy"), physics)
    np.save(os.path.join(args.output_dir, "instrument_latents.npy"), instrument)

    chip_label = np.asarray([f"s{patch.sector[r]}-c{patch.camera[r]}-{patch.ccd[r]}"
                             for r in rows])
    table = pd.DataFrame({
        "row": rows, "TIC": patch.tic[rows], "split": splits, "chip": chip_label,
        "sector": patch.sector[rows], "camera": patch.camera[rows], "ccd": patch.ccd[rows],
        "detector_x": patch.det_x[rows], "detector_y": patch.det_y[rows],
        "valid_cadences": patch.n_valid[rows],
        "gap_fraction": 1.0 - patch.n_valid[rows] / patch.curve_length})
    table.to_csv(os.path.join(args.output_dir, "selected_anchors.csv"), index=False)

    is_train = splits == "train"
    is_val = splits == "val"
    print("fitting scalers and PCA (train only)", flush=True)
    prepared, dims, retained, curves_ev = {}, {}, {}, {}
    for label, latents in (("physics", physics), ("instrument", instrument)):
        values, n, var, cumulative = fit_pca(latents, is_train, args.seed, label,
                                             args.output_dir)
        prepared[label], dims[label], retained[label] = values, n, var
        curves_ev[label] = cumulative

    results, flow_reports = {}, {}
    for label in ("physics", "instrument"):
        values = prepared[label]
        print(f"fitting {label} flow ({values.shape[1]}D, batch {BATCH_SIZE})", flush=True)
        flow, best, history = fit_flow(values[is_train], values[is_val], values.shape[1],
                                       args.seed, label, batch_size=BATCH_SIZE)
        history.to_csv(os.path.join(args.output_dir, f"{label}_flow_history.csv"),
                       index=False)
        torch.save({"state_dict": flow.state_dict(), "features": values.shape[1],
                    "best_epoch": best["epoch"], "val_nll": best["val_nll"]},
                   os.path.join(args.output_dir, f"{label}_flow.pt"))
        scores = nll(flow, values)
        assert np.isfinite(scores).all(), f"{label}: non-finite NLL"
        results[label] = scores
        flow_reports[label] = {"best_epoch": best["epoch"],
                               "best_train_nll": best["train_nll"],
                               "best_val_nll": best["val_nll"],
                               "pca_dim": int(values.shape[1]),
                               "retained_variance": retained[label]}
        table[f"{label}_nll"] = scores
        table[f"{label}_percentile"] = percentile_against(scores, scores[is_train])

    def classify(record):
        p = record["physics_percentile"] >= THRESHOLD
        i = record["instrument_percentile"] >= THRESHOLD
        return "both" if p and i else "physics" if p else "instrument" if i else "typical"
    table["classification"] = table.apply(classify, axis=1)
    for label in ("physics", "instrument"):
        test = table["split"] == "test"
        table[f"{label}_rank"] = np.nan
        table.loc[test, f"{label}_rank"] = table.loc[test, f"{label}_nll"].rank(
            ascending=False).astype(int)
    table.to_csv(os.path.join(args.output_dir, "anomaly_scores.csv"), index=False)

    # ------------------------------------------------------- confounder checks
    confounders = {}
    for label in ("physics", "instrument"):
        confounders[label] = {
            "spearman_vs_valid_cadences": safe_spearman(table[f"{label}_nll"], table["valid_cadences"]),
            "spearman_vs_gap_fraction": safe_spearman(table[f"{label}_nll"], table["gap_fraction"]),
            "spearman_vs_sector": safe_spearman(table[f"{label}_nll"], table["sector"]),
            "spearman_vs_chip_index": safe_spearman(table[f"{label}_nll"], pd.factorize(table["chip"])[0]),
            "chip_median_nll_spread": float(
                table.groupby("chip")[f"{label}_nll"].median().max()
                - table.groupby("chip")[f"{label}_nll"].median().min()),
        }

    grouped = table.groupby("chip")
    per_chip = pd.DataFrame({
        "chip": grouped.size().index, "n": grouped.size().to_numpy(),
        "physics": grouped.apply(lambda g: int((g["physics_percentile"] >= THRESHOLD).sum()),
                                 include_groups=False).to_numpy(),
        "instrument": grouped.apply(
            lambda g: int((g["instrument_percentile"] >= THRESHOLD).sum()),
            include_groups=False).to_numpy()})
    per_chip["physics_rate"] = per_chip["physics"] / per_chip["n"]
    per_chip["instrument_rate"] = per_chip["instrument"] / per_chip["n"]
    per_chip.to_csv(os.path.join(args.output_dir, "anomalies_by_chip.csv"), index=False)

    warnings_list = []
    for label in ("physics", "instrument"):
        total = int(per_chip[label].sum())
        if total:
            share = per_chip[label] / total
            worst = per_chip.loc[share.idxmax()]
            expected = 1.0 / len(per_chip)
            if float(share.max()) > 3 * expected:
                warnings_list.append(
                    f"{label}: chip {worst['chip']} holds {float(share.max()):.1%} of "
                    f"candidates ({expected:.1%} expected) -- the flow may be reading "
                    f"chip identity rather than within-chip behaviour")

    # ------------------------------------------------------------------- plots
    pca_curve_plot(curves_ev, dims, os.path.join(args.output_dir, "pca_explained_variance.png"))
    nll_plot(table, os.path.join(args.output_dir, "nll_distributions.png"))
    comparison_plot(table, os.path.join(args.output_dir, "anomaly_score_comparison.png"))
    chip_plot(per_chip, os.path.join(args.output_dir, "anomalies_by_chip.png"))

    rng = np.random.default_rng(args.seed)
    subset = (np.sort(rng.permutation(len(table))[:UMAP_MAX]) if len(table) > UMAP_MAX
              else np.arange(len(table)))
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, latents in (("physics", physics), ("instrument", instrument)):
            coords = np.asarray(umap.UMAP(**UMAP_KWARGS).fit_transform(
                StandardScaler().fit_transform(latents[subset])))
            umap_plot(coords, table[f"{label}_percentile"].to_numpy()[subset],
                      f"{label} latent UMAP ({len(subset)} anchors)",
                      os.path.join(args.output_dir, f"{label}_umap_anomaly_scores.png"))

    # -------------------------------------------------------- candidate galleries
    quiet = None
    reference_path = os.path.join(run_dir, "reference_context.pt")
    if os.path.exists(reference_path):
        quiet = load_reference_context(run_dir)
    else:
        print("reference_context.pt missing: galleries show raw curves only", flush=True)

    test_table = table[table["split"] == "test"]
    selections = {
        "physics": test_table[test_table["classification"] == "physics"].nlargest(
            12, "physics_nll"),
        "instrument": test_table[test_table["classification"] == "instrument"].nlargest(
            12, "instrument_nll"),
        "both": test_table[test_table["classification"] == "both"].nlargest(
            12, "physics_nll")}
    needed = sorted({int(r) for frame in selections.values() for r in frame["row"]})
    gallery_curves = {}
    if needed and quiet is not None:
        # The quiet reference lives on one chip's cadence grid; only score anchors from
        # that chip against it, otherwise the contexts are not comparable.
        sector0 = patch.target[0]
        usable = [r for r in needed if patch.sector[r] == sector0]
        if usable:
            peers = np.stack([patch.peers_for_row(int(r), "test")[0] for r in usable])
            actual, ref, _, _, _ = dual_context_prediction(
                model, torch.from_numpy(patch.X[usable]), torch.from_numpy(patch.M[usable]),
                torch.from_numpy(patch.X[peers]), torch.from_numpy(patch.M[peers]),
                quiet["peer_raw"].unsqueeze(0).expand(len(usable), -1, -1),
                quiet["peer_mask"].unsqueeze(0).expand(len(usable), -1, -1), masks, device)
            correction = (actual - ref).numpy()
            for k, row in enumerate(usable):
                peer_flux, peer_mask = patch.X[peers[k]], patch.M[peers[k]]
                seen = peer_mask.sum(axis=0) > 0
                common = np.zeros(patch.curve_length)
                common[seen] = np.nanmedian(
                    np.where(peer_mask, peer_flux, np.nan)[:, seen], axis=0)
                gallery_curves[row] = {
                    "reconstructed": actual[k].numpy(),
                    "cleaned": patch.X[row] - correction[k],
                    "correction": correction[k],
                    "peer_common": np.nan_to_num(common)}

    for kind, frame in selections.items():
        entries = [(int(r["row"]),
                    f"TIC {r['TIC']}\n{r['chip']}\nphy {r['physics_percentile']:.3f}\n"
                    f"ins {r['instrument_percentile']:.3f}")
                   for _, r in frame.iterrows()]
        gallery(patch, entries, gallery_curves,
                f"top {kind} candidates (test split)",
                os.path.join(args.output_dir, f"top_{kind}_candidates_test.png"))

    # -------------------------------------------------- comparison with the old run
    comparison = None
    if args.old_scores and os.path.exists(args.old_scores):
        old = pd.read_csv(args.old_scores)
        old["TIC"] = old["TIC"].astype(str)          # CSV round-trip makes TIC int64
        merged = table.assign(TIC=table["TIC"].astype(str)).merge(
            old, on="TIC", suffixes=("_new", "_old"))
        if len(merged):
            old_physics = set(old.loc[old["physics_percentile"] >= THRESHOLD, "TIC"].astype(str))
            new_physics = set(table.loc[table["physics_percentile"] >= THRESHOLD, "TIC"].astype(str))
            old_instrument = set(old.loc[old["instrument_percentile"] >= THRESHOLD, "TIC"].astype(str))
            new_instrument = set(table.loc[table["instrument_percentile"] >= THRESHOLD, "TIC"].astype(str))

            def jaccard(a, b):
                return len(a & b) / len(a | b) if (a | b) else 0.0
            comparison = {
                "shared_anchors": int(len(merged)),
                "spearman_physics_old_new": safe_spearman(merged["physics_percentile_old"], merged["physics_percentile_new"]),
                "spearman_instrument_old_new": safe_spearman(merged["instrument_percentile_old"], merged["instrument_percentile_new"]),
                "physics_overlap": len(old_physics & new_physics),
                "physics_jaccard": jaccard(old_physics, new_physics),
                "instrument_overlap": len(old_instrument & new_instrument),
                "instrument_jaccard": jaccard(old_instrument, new_instrument),
                "old_counts": old["classification"].value_counts().to_dict(),
                "new_counts": table["classification"].value_counts().to_dict(),
            }

    both_count = int((table["classification"] == "both").sum())
    n_physics = int((table["physics_percentile"] >= THRESHOLD).sum())
    n_instrument = int((table["instrument_percentile"] >= THRESHOLD).sum())
    expected_both = n_physics * n_instrument / max(len(table), 1)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "n_anchors": int(len(table)), "counts_by_split": counts,
        "n_chips": len(patch.chips), "shortfalls": shortfalls,
        "pca": {label: {"dim": dims[label], "retained_variance": retained[label]}
                for label in dims},
        "flows": flow_reports,
        "nll_summary": {label: {split: {
            "median": float(table.loc[table["split"] == split, f"{label}_nll"].median()),
            "q1": float(table.loc[table["split"] == split, f"{label}_nll"].quantile(0.25)),
            "q3": float(table.loc[table["split"] == split, f"{label}_nll"].quantile(0.75))}
            for split in PER_CHIP} for label in dims},
        "classification_counts": table["classification"].value_counts().to_dict(),
        "classification_counts_test": table.loc[table["split"] == "test",
                                                "classification"].value_counts().to_dict(),
        "both_observed": both_count, "both_expected_if_independent": expected_both,
        "both_excess_over_chance": both_count / expected_both if expected_both else None,
        "confounders": confounders,
        "chip_concentration_warnings": warnings_list,
        "comparison_with_old": comparison,
        "checks": {"all_scores_finite": True, "weights_unchanged": bool(weights_unchanged),
                   "fit_on_train_only": True, "model_in_eval_mode": bool(not model.training),
                   "reference_context_available": quiet is not None},
        "runtime_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    # ------------------------------------------------------------------- report
    print(f"\n=== sampling ===\n  {len(table)} anchors over {len(patch.chips)} chips: {counts}")
    if shortfalls:
        print(f"  shortfalls: {len(shortfalls)} chip/split combinations under target")
    print("\n=== PCA ===")
    for label in dims:
        print(f"  {label:11s} 512 -> {dims[label]:3d} components, retained "
              f"{retained[label]:.4f}")
    print("\n=== flows ===")
    for label, report in flow_reports.items():
        print(f"  {label:11s} best epoch {report['best_epoch']:2d}  train NLL "
              f"{report['best_train_nll']:9.3f}  val NLL {report['best_val_nll']:9.3f}")
    print("\n=== NLL by split ===")
    for label in dims:
        for split in ("train", "val", "test"):
            values = summary["nll_summary"][label][split]
            print(f"  {label:11s} {split:5s} median {values['median']:9.3f}  IQR "
                  f"[{values['q1']:9.3f}, {values['q3']:9.3f}]")
    print(f"\n=== classification ===\n  all:  {summary['classification_counts']}")
    print(f"  test: {summary['classification_counts_test']}")
    print(f"  both observed {both_count}, expected {expected_both:.1f} under "
          f"independence ({summary['both_excess_over_chance']:.2f}x)")
    print("\n=== confounders (Spearman of NLL vs) ===")
    for label, values in confounders.items():
        print(f"  {label:11s} valid_cadences {values['spearman_vs_valid_cadences']:+.3f}  "
              f"gap {values['spearman_vs_gap_fraction']:+.3f}  "
              f"sector {values['spearman_vs_sector']:+.3f}  "
              f"chip {values['spearman_vs_chip_index']:+.3f}")
    print("\n=== per-chip candidate rate ===")
    print(f"  physics    min {per_chip['physics_rate'].min():.3f}  median "
          f"{per_chip['physics_rate'].median():.3f}  max {per_chip['physics_rate'].max():.3f}"
          f"  (worst chip {per_chip.loc[per_chip['physics_rate'].idxmax(), 'chip']})")
    print(f"  instrument min {per_chip['instrument_rate'].min():.3f}  median "
          f"{per_chip['instrument_rate'].median():.3f}  max "
          f"{per_chip['instrument_rate'].max():.3f}"
          f"  (worst chip {per_chip.loc[per_chip['instrument_rate'].idxmax(), 'chip']})")
    for message in warnings_list:
        print(f"  WARNING: {message}")
    if comparison:
        print("\n=== versus the 1000-anchor analysis ===")
        print(f"  shared anchors {comparison['shared_anchors']}")
        print(f"  Spearman old/new  physics {comparison['spearman_physics_old_new']:+.3f}"
              f"  instrument {comparison['spearman_instrument_old_new']:+.3f}")
        print(f"  candidate Jaccard physics {comparison['physics_jaccard']:.3f}"
              f"  instrument {comparison['instrument_jaccard']:.3f}")
        print(f"  old counts {comparison['old_counts']}")
        print(f"  new counts {comparison['new_counts']}")
    print(f"\noutputs in {args.output_dir}\nruntime: {summary['runtime_seconds']}s")


if __name__ == "__main__":
    main()
