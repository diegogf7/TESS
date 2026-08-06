"""MOMENT foundation-model anomaly baseline: does a general embedding separate
physics-like from instrument-like anomalies, the way the disentangled latents do?

MOMENT sees one complete observed anchor curve and returns one general-purpose
embedding. The SAME PCA + normalizing-flow anomaly procedure used on the disentangled
latents is then applied to it, so the only thing that differs is the representation.

Frozen throughout: MOMENT weights, and the disentanglement checkpoint. Only a scaler,
a PCA and one flow are fitted, on the training split alone.

MOMENT-1-small takes 512 points, so a 1024-cadence curve is split into [0:512] and
[512:1024] and the two embeddings are CONCATENATED in chronological order. Averaging
them would erase which half contained an event.

    ~/tess-venv/bin/python -m disentangle_attempt.moment_anomaly_baseline \
      --checkpoint disentangle_attempt/outputs/fast_strict/best.pt \
      --latents-dir disentangle_attempt/outputs/fast_strict/umaps \
      --disentangled-scores disentangle_attempt/outputs/fast_strict/anomaly_flows/anomaly_scores.csv
"""

import argparse
import hashlib
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

from disentangle_attempt.dataset import (CrossSectorPatch,
                                        infer_require_cross_sector)
from disentangle_attempt.fit_anomaly_flows import (THRESHOLD, fit_flow, nll,
                                                   percentile_against)
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

MOMENT_NAME = "AutonLab/MOMENT-1-small"
WINDOW = 512
PCA_DIM = 32
UMAP_KWARGS = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=42)


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def select_anchors(patch, max_samples):
    """Reproduce plot_latent_umaps' selection exactly, so rows line up 1:1."""
    rows, splits = [], []
    for split in ("train", "val", "test"):
        for anchor in patch.split_anchors[split]:
            rows.append(int(anchor))
            splits.append(split)
    rows = np.asarray(rows)
    if len(rows) > max_samples:
        pick = np.linspace(0, len(rows) - 1, max_samples).astype(int)
        rows = rows[pick]
        splits = [splits[i] for i in pick]
    return rows, splits


@torch.no_grad()
def moment_embeddings(patch, rows, batch=32):
    """[N, 1024]: two chronological 512-cadence windows, concatenated.

    Invalid cadences enter as numerical zero with mask 0 -- padding, not observations.
    """
    from momentfm import MOMENTPipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MOMENTPipeline.from_pretrained(
            MOMENT_NAME, model_kwargs={"task_name": "embedding"})
        model.init()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    before = state_hash(model)

    pieces = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        flux = torch.from_numpy(patch.X[chunk]).float()
        mask = torch.from_numpy(patch.M[chunk]).float()
        halves = []
        for lo in range(0, patch.curve_length, WINDOW):
            window = flux[:, lo:lo + WINDOW].unsqueeze(1)          # [B, 1, 512]
            window_mask = mask[:, lo:lo + WINDOW]                  # [B, 512]
            output = model(x_enc=window, input_mask=window_mask)
            halves.append(output.embeddings.float())
        pieces.append(torch.cat(halves, dim=-1).cpu().numpy())     # chronological
    embeddings = np.concatenate(pieces).astype(np.float32)
    assert state_hash(model) == before, "MOMENT weights changed during extraction"
    return embeddings, before


# ------------------------------------------------------------------------- plots
def umap_plot(coords, percentile, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = np.argsort(percentile)
    art = ax.scatter(coords[order, 0], coords[order, 1], c=percentile[order], s=7,
                     cmap="viridis", vmin=0, vmax=1, linewidths=0)
    high = percentile >= THRESHOLD
    ax.scatter(coords[high, 0], coords[high, 1], s=45, facecolors="none",
               edgecolors="red", linewidths=0.8, label=f"percentile >= {THRESHOLD}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("MOMENT embedding UMAP, coloured by MOMENT anomaly percentile\n"
                 "(UMAP distance is visualization only, never the score)", fontsize=10)
    ax.legend(fontsize=8)
    fig.colorbar(art, ax=ax, fraction=0.04, label="MOMENT anomaly percentile")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scatter_vs_disentangled(table, path):
    colours = {"physics": "tab:red", "instrument": "tab:blue", "both": "tab:purple",
               "typical": "0.7"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, column, label in ((axes[0], "physics_percentile", "physics"),
                              (axes[1], "instrument_percentile", "instrument")):
        for name, colour in colours.items():
            pick = table["classification"] == name
            ax.scatter(table.loc[pick, column], table.loc[pick, "moment_percentile"],
                       s=8, color=colour, label=name, linewidths=0)
        ax.axvline(THRESHOLD, color="0.3", lw=0.8, ls="--")
        ax.axhline(THRESHOLD, color="0.3", lw=0.8, ls="--")
        ax.set_xlabel(f"{label} anomaly percentile")
        ax.set_ylabel("MOMENT anomaly percentile")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("MOMENT versus disentangled anomaly percentiles", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def nll_plot(table, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, colour in (("train", "0.6"), ("val", "tab:orange"), ("test", "tab:blue")):
        values = table.loc[table["split"] == split, "moment_nll"]
        if len(values):
            ax.hist(values, bins=40, histtype="step", color=colour, label=split,
                    density=True)
    ax.set_xlabel("MOMENT negative log-likelihood")
    ax.set_title("MOMENT NLL by split (percentiles are ~uniform on train by "
                 "construction, so NLL is plotted here)", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def curve_gallery(patch, entries, curves, title, path, ncol_note=None):
    """entries: list of (row, label_text) or empty."""
    if not entries:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"{title}: no examples in this category", ha="center")
        ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(entries), 1, figsize=(11, 1.9 * len(entries)))
    axes = np.atleast_1d(axes)
    for ax, (row, label) in zip(axes, entries):
        valid = patch.M[row]
        x = np.arange(len(valid))
        ax.plot(x, np.where(valid, patch.X[row], np.nan), lw=0.7, color="0.6", label="raw")
        if row in curves:
            ax.plot(x, np.where(valid, curves[row]["cleaned"], np.nan), lw=0.8,
                    color="tab:blue", label="cleaned (disentangled)")
        ax.set_ylabel(label, fontsize=6)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    fig.suptitle(title + (f"\n{ncol_note}" if ncol_note else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def overlap_stats(a, b):
    a, b = set(a), set(b)
    union = len(a | b)
    return {"n_a": len(a), "n_b": len(b), "overlap": len(a & b),
            "jaccard": (len(a & b) / union) if union else 0.0}


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--disentangled-scores", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.disentangled_scores)), "moment_baseline")
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    target = state.get("target") or ("auto", "auto", "auto")
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=target[0], camera=target[1], ccd=target[2],
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(
            config, args.require_cross_sector), verbose=False)

    rows, splits = select_anchors(patch, args.max_samples)
    reference = pd.read_csv(os.path.join(args.latents_dir, "umap_coordinates.csv"))
    assert len(rows) == len(reference), "anchor count differs from the UMAP metadata"
    assert list(patch.tic[rows]) == [str(t) for t in reference["TIC"]], \
        "anchor order differs from the existing analysis"
    assert splits == list(reference["split"]), "split assignment differs"
    assert len(set(rows)) == len(rows), "an anchor row repeats"
    print(f"{len(rows)} anchors, identical rows and splits to the disentangled analysis",
          flush=True)

    print(f"extracting {MOMENT_NAME} embeddings (frozen)", flush=True)
    embeddings, weight_hash = moment_embeddings(patch, rows)
    assert np.isfinite(embeddings).all(), "non-finite MOMENT embeddings"
    np.save(os.path.join(out_dir, "moment_embeddings.npy"), embeddings)
    metadata = pd.DataFrame({
        "row": rows, "TIC": patch.tic[rows], "split": splits,
        "sector": patch.sector[rows], "camera": patch.camera[rows], "ccd": patch.ccd[rows],
        "detector_x": patch.det_x[rows], "detector_y": patch.det_y[rows]})
    metadata.to_csv(os.path.join(out_dir, "moment_metadata.csv"), index=False)
    print(f"  embeddings {embeddings.shape} (2 x 512-cadence windows concatenated)",
          flush=True)

    # ------------------------------------------------- train-only scaler/PCA/flow
    is_train = np.asarray(splits) == "train"
    is_val = np.asarray(splits) == "val"
    scaler = StandardScaler().fit(embeddings[is_train])
    pca = PCA(n_components=PCA_DIM, random_state=args.seed).fit(
        scaler.transform(embeddings[is_train]))
    values = pca.transform(scaler.transform(embeddings))
    joblib.dump(scaler, os.path.join(out_dir, "moment_scaler.joblib"))
    joblib.dump(pca, os.path.join(out_dir, "moment_pca.joblib"))
    retained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {embeddings.shape[1]} -> {PCA_DIM}D, retained variance {retained:.3f}",
          flush=True)

    flow, best, history = fit_flow(values[is_train], values[is_val], PCA_DIM,
                                   args.seed, "moment")
    history.to_csv(os.path.join(out_dir, "moment_flow_history.csv"), index=False)
    torch.save({"state_dict": flow.state_dict(), "features": PCA_DIM,
                "best_epoch": best["epoch"], "val_nll": best["val_nll"]},
               os.path.join(out_dir, "moment_flow.pt"))

    scores = nll(flow, values)
    assert np.isfinite(scores).all(), "non-finite MOMENT NLL"
    percentile = percentile_against(scores, scores[is_train])

    # ------------------------------------------------------------------ the join
    existing = pd.read_csv(args.disentangled_scores)
    assert len(existing) == len(rows), "disentangled score table has a different length"
    assert list(existing["TIC"].astype(str)) == list(patch.tic[rows]), \
        "disentangled scores are not in the same observation order"
    table = metadata.copy()
    table["moment_nll"] = scores
    table["moment_percentile"] = percentile
    table["moment_anomaly"] = percentile >= THRESHOLD
    for column in ("physics_nll", "instrument_nll", "physics_percentile",
                   "instrument_percentile", "classification"):
        table[column] = existing[column].to_numpy()
    table.to_csv(os.path.join(out_dir, "moment_anomaly_scores.csv"), index=False)

    # --------------------------------------------------------------- comparisons
    comparisons = {}
    for label in ("all", "train", "val", "test"):
        sub = table if label == "all" else table[table["split"] == label]
        moment_set = set(sub.loc[sub["moment_anomaly"], "row"])
        physics_set = set(sub.loc[sub["physics_percentile"] >= THRESHOLD, "row"])
        instrument_set = set(sub.loc[sub["instrument_percentile"] >= THRESHOLD, "row"])
        counts = sub["classification"].value_counts().to_dict()
        spear_p = spearmanr(sub["moment_percentile"], sub["physics_percentile"])[0] \
            if len(sub) > 2 else np.nan
        spear_i = spearmanr(sub["moment_percentile"], sub["instrument_percentile"])[0] \
            if len(sub) > 2 else np.nan
        comparisons[label] = {
            "n": int(len(sub)),
            "n_moment_anomalies": len(moment_set),
            "n_disentangled_physics": int(counts.get("physics", 0)),
            "n_disentangled_instrument": int(counts.get("instrument", 0)),
            "n_disentangled_both": int(counts.get("both", 0)),
            "n_disentangled_typical": int(counts.get("typical", 0)),
            "moment_vs_physics": overlap_stats(moment_set, physics_set),
            "moment_vs_instrument": overlap_stats(moment_set, instrument_set),
            "moment_anomalies_by_disentangled_category":
                sub.loc[sub["moment_anomaly"], "classification"].value_counts().to_dict(),
            "spearman_moment_physics": float(spear_p),
            "spearman_moment_instrument": float(spear_i),
        }

    # --------------------------------------------------------------- galleries
    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    quiet = load_reference_context(run_dir, expected_cadence_ids=patch.grids[patch.target[0]])
    masks = complementary_masks(config["curve_length"], n_masks=4)

    test = table[table["split"] == "test"]
    top_moment = test.nlargest(min(12, len(test)), "moment_nll")
    categories = {
        "MOMENT and physics": test[(test["moment_anomaly"])
                                   & (test["physics_percentile"] >= THRESHOLD)],
        "MOMENT and instrument": test[(test["moment_anomaly"])
                                      & (test["instrument_percentile"] >= THRESHOLD)],
        "MOMENT only": test[(test["moment_anomaly"])
                            & (test["physics_percentile"] < THRESHOLD)
                            & (test["instrument_percentile"] < THRESHOLD)],
        "physics only": test[(~test["moment_anomaly"])
                             & (test["physics_percentile"] >= THRESHOLD)],
    }
    needed = sorted(set(top_moment["row"]).union(
        *[set(frame.nlargest(min(3, len(frame)), "moment_nll")["row"])
          for frame in categories.values()] or [set()]))
    curves = {}
    if needed:
        peer_rows = np.stack([patch.peers_for_row(int(r), "test")[0] for r in needed])
        actual, ref, _, _, _ = dual_context_prediction(
            model, torch.from_numpy(patch.X[needed]), torch.from_numpy(patch.M[needed]),
            torch.from_numpy(patch.X[peer_rows]), torch.from_numpy(patch.M[peer_rows]),
            quiet["peer_raw"].unsqueeze(0).expand(len(needed), -1, -1),
            quiet["peer_mask"].unsqueeze(0).expand(len(needed), -1, -1), masks, device)
        correction = (actual - ref).numpy()
        for k, row in enumerate(needed):
            curves[row] = {"cleaned": patch.X[row] - correction[k]}

    def label_of(record):
        return (f"TIC {record['TIC']}\nMOM {record['moment_percentile']:.3f}\n"
                f"phy {record['physics_percentile']:.3f}\n"
                f"ins {record['instrument_percentile']:.3f}")

    curve_gallery(patch, [(int(r["row"]), label_of(r)) for _, r in top_moment.iterrows()],
                  curves, "top MOMENT candidates (held-out test split)",
                  os.path.join(out_dir, "top_moment_candidates_test.png"))

    entries = []
    for name, frame in categories.items():
        picked = frame.nlargest(min(3, len(frame)), "moment_nll")
        for _, record in picked.iterrows():
            entries.append((int(record["row"]), f"[{name}]\n" + label_of(record)))
    empty = [name for name, frame in categories.items() if frame.empty]
    curve_gallery(patch, entries, curves, "candidate overlap examples (test split)",
                  os.path.join(out_dir, "candidate_overlap_examples.png"),
                  ncol_note=("empty categories: " + ", ".join(empty)) if empty else None)

    # ------------------------------------------------------------------- plots
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = np.asarray(umap.UMAP(**UMAP_KWARGS).fit_transform(
            StandardScaler().fit_transform(embeddings)))
    umap_plot(coords, percentile, os.path.join(out_dir, "moment_umap_anomaly_scores.png"))
    scatter_vs_disentangled(table, os.path.join(out_dir, "moment_vs_disentangled.png"))
    nll_plot(table, os.path.join(out_dir, "nll_comparison.png"))

    is_test = table["split"] == "test"
    summary = {
        "moment_model": MOMENT_NAME,
        "moment_weight_hash": weight_hash,
        "moment_weights_frozen": True,
        "n_embeddings": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "windowing": "two 512-cadence windows concatenated chronologically",
        "pca_dim": PCA_DIM, "pca_retained_variance": retained,
        "flow": {"best_epoch": best["epoch"], "best_train_nll": best["train_nll"],
                 "best_val_nll": best["val_nll"]},
        "test_nll": {"median": float(table.loc[is_test, "moment_nll"].median()),
                     "q1": float(table.loc[is_test, "moment_nll"].quantile(0.25)),
                     "q3": float(table.loc[is_test, "moment_nll"].quantile(0.75))},
        "comparisons": comparisons,
        "empty_categories": empty,
        "checks": {"same_rows_and_splits": True, "all_embeddings_finite": True,
                   "all_scores_finite": True,
                   "scaler_pca_flow_fit_on_train_only": True},
        "interpretation_note": ("Candidates are tail-of-distribution picks, not "
                                "confirmed anomalies. UMAP is visualization only."),
        "runtime_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print(f"\nMOMENT model: {MOMENT_NAME} (frozen, weight hash {weight_hash})")
    print(f"embeddings: {embeddings.shape}   PCA retained variance {retained:.3f}")
    print(f"flow: best epoch {best['epoch']}  train NLL {best['train_nll']:.3f}  "
          f"val NLL {best['val_nll']:.3f}")
    t = summary["test_nll"]
    print(f"test MOMENT NLL: median {t['median']:.3f}  IQR [{t['q1']:.3f}, {t['q3']:.3f}]")
    print("\n=== candidate counts and overlap ===")
    print(f"{'split':6s} {'n':>5s} {'MOM':>5s} {'phy':>5s} {'ins':>5s} {'both':>5s} "
          f"{'MOM&phy':>8s} {'J':>6s} {'MOM&ins':>8s} {'J':>6s} {'rho_phy':>8s} {'rho_ins':>8s}")
    for label, values in comparisons.items():
        mp, mi = values["moment_vs_physics"], values["moment_vs_instrument"]
        print(f"{label:6s} {values['n']:5d} {values['n_moment_anomalies']:5d} "
              f"{values['n_disentangled_physics']:5d} {values['n_disentangled_instrument']:5d} "
              f"{values['n_disentangled_both']:5d} {mp['overlap']:8d} {mp['jaccard']:6.3f} "
              f"{mi['overlap']:8d} {mi['jaccard']:6.3f} "
              f"{values['spearman_moment_physics']:8.3f} "
              f"{values['spearman_moment_instrument']:8.3f}")
    print("\nMOMENT anomalies by disentangled category (all): "
          f"{comparisons['all']['moment_anomalies_by_disentangled_category']}")
    if empty:
        print(f"empty overlap categories in test: {', '.join(empty)}")
    print(f"\nfiles in {out_dir}:")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {name}")
    print(f"runtime: {summary['runtime_seconds']}s")


if __name__ == "__main__":
    main()
