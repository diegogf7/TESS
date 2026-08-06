"""Is the instrument anomaly score reading missing cadences rather than flux?

The instrument latent is built from eight peers via masked token pooling, and a fully
masked block pools to exactly zero -- so a peer group's GAP PATTERN is written into the
latent whether or not its flux is unusual. Chips differ systematically in coverage, so
a global density over that latent could be scoring "this chip has unusual gaps".

Three tests, strongest last:

1. correlation of instrument NLL with gap features, globally and WITHIN chip (a global
   correlation can be entirely between-chip structure);
2. how well gap features alone predict the NLL (ridge R^2, train/test split by chip);
3. a MASK-ONLY control -- re-encode the same peers with their flux replaced by one
   shared noise realization, keeping every mask identical, then run the same
   PCA + flow + percentile pipeline. If the mask-only ranking reproduces the real
   candidate set, the score is about gaps, not about flux.

Nothing is retrained; the disentanglement model stays frozen.

    python -m disentangle_attempt.diagnose_instrument_gaps \
      --checkpoint .../multichip_5sectors_v1/best.pt \
      --scores .../anomaly_analysis_20k_pca90/anomaly_scores.csv \
      --parquet ... --output-dir .../anomaly_analysis_20k_pca90/gap_diagnostic
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from disentangle_attempt.dataset import (CrossSectorPatch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.fit_anomaly_flows import THRESHOLD, fit_flow, nll, percentile_against
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.plot_latent_umaps import instrument_latents
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

MIN_VARIANCE = 0.90
BATCH_SIZE = 256


def gap_features(patch, peer_rows):
    """Per-anchor summary of the peer group's missing-cadence structure."""
    masks = patch.M[peer_rows]                       # [N, 8, L]
    per_peer = masks.mean(axis=2)                    # coverage of each peer
    union = masks.any(axis=1).mean(axis=1)
    intersection = masks.all(axis=1).mean(axis=1)
    counts = masks.sum(axis=1)                       # peers observing each cadence
    runs = np.diff((counts == 0).astype(np.int8), axis=1)
    return pd.DataFrame({
        "peer_valid_mean": per_peer.mean(axis=1),
        "peer_valid_min": per_peer.min(axis=1),
        "peer_valid_std": per_peer.std(axis=1),
        "union_coverage": union,
        "intersection_coverage": intersection,
        "fully_empty_fraction": (counts == 0).mean(axis=1),
        "n_gap_runs": (runs == 1).sum(axis=1).astype(float),
    })


@torch.no_grad()
def mask_only_latents(model, patch, peer_rows, device, seed, batch=32):
    """Same masks, same encoder -- flux replaced by ONE shared noise realization.

    Every anchor therefore sees identical 'flux'; only the mask differs, so whatever
    structure survives is gap structure.
    """
    generator = torch.Generator().manual_seed(seed)
    surrogate = torch.randn(patch.curve_length, generator=generator)
    out = np.zeros((len(peer_rows), model.latent_size), dtype=np.float32)
    for start in range(0, len(peer_rows), batch):
        chunk = peer_rows[start:start + batch]
        mask = torch.from_numpy(patch.M[chunk]).to(device)
        flux = surrogate.to(device).expand(mask.shape).clone()
        flux = flux.masked_fill(~mask, 0.0)          # gaps stay normalized zero
        tokens, _ = model.encode_peers(flux, mask)
        out[start:start + len(chunk)] = tokens.flatten(2).mean(1).cpu().numpy()
    return out


def score_pipeline(latents, is_train, is_val, seed, label):
    """The exact PCA + flow + percentile pipeline used by the real analysis."""
    scaler = StandardScaler().fit(latents[is_train])
    scaled = scaler.transform(latents)
    full = PCA(random_state=seed).fit(scaled[is_train])
    n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), MIN_VARIANCE) + 1)
    pca = PCA(n_components=n, random_state=seed).fit(scaled[is_train])
    values = pca.transform(scaled)
    flow, best, _ = fit_flow(values[is_train], values[is_val], n, seed, label,
                             batch_size=BATCH_SIZE)
    scores = nll(flow, values)
    return scores, percentile_against(scores, scores[is_train]), n, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    table = pd.read_csv(args.scores)
    rows = table["row"].to_numpy()
    splits = table["split"].to_numpy()

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

    index_of = {split: {int(a): i for i, a in enumerate(patch.split_anchors[split])}
                for split in ("train", "val", "test")}
    peer_rows = np.stack([patch.peers[s][0][index_of[s][int(r)]]
                          for r, s in zip(rows, splits)])

    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    features = gap_features(patch, peer_rows)
    features.index = table.index
    is_train = splits == "train"
    is_val = splits == "val"

    # ---- test 1: correlation, global and within chip ------------------------------
    global_corr = {c: float(spearmanr(table["instrument_nll"], features[c])[0])
                   for c in features.columns}
    within = []
    for chip, group in table.groupby("chip"):
        if len(group) < 30:
            continue
        sub = features.loc[group.index]
        within.append({c: float(spearmanr(group["instrument_nll"], sub[c])[0])
                       for c in features.columns})
    within_median = {c: float(np.nanmedian([w[c] for w in within])) for c in features.columns}

    # ---- test 2: how much of the NLL do gaps alone explain? -----------------------
    scaler = StandardScaler().fit(features[is_train])
    ridge = Ridge(alpha=1.0).fit(scaler.transform(features[is_train]),
                                 table.loc[is_train, "instrument_nll"])
    predicted = ridge.predict(scaler.transform(features))
    held = ~is_train
    ss_res = float(((table.loc[held, "instrument_nll"] - predicted[held]) ** 2).sum())
    ss_tot = float(((table.loc[held, "instrument_nll"]
                     - table.loc[held, "instrument_nll"].mean()) ** 2).sum())
    ridge_r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    # ---- test 3: mask-only control ------------------------------------------------
    print("encoding mask-only surrogate peers", flush=True)
    surrogate = mask_only_latents(model, patch, peer_rows, device, args.seed)
    scores, percentile, n_components, best = score_pipeline(
        surrogate, is_train, is_val, args.seed, "mask_only")
    table["mask_only_nll"] = scores
    table["mask_only_percentile"] = percentile

    # Degeneracy guard: if every anchor's peers share essentially the same mask, the
    # surrogate latents are identical, every percentile ties at 1.0, and "recall" is a
    # meaningless 1.0. That is the single-chip case, not evidence about gaps.
    surrogate_spread = float(np.mean(np.std(surrogate, axis=0)))
    real_set = set(table.loc[table["instrument_percentile"] >= THRESHOLD, "row"])
    mask_set = set(table.loc[table["mask_only_percentile"] >= THRESHOLD, "row"])
    union = len(real_set | mask_set)
    agreement = {
        "spearman_real_vs_mask_only": float(spearmanr(table["instrument_percentile"],
                                                      table["mask_only_percentile"])[0]),
        "real_candidates": len(real_set), "mask_only_candidates": len(mask_set),
        "overlap": len(real_set & mask_set),
        "jaccard": (len(real_set & mask_set) / union) if union else 0.0,
        "recall_of_real_by_mask_only": (len(real_set & mask_set) / len(real_set)
                                        if real_set else float("nan")),
        "mask_only_pca_dim": n_components,
        "mask_only_latent_spread": surrogate_spread,
        "mask_only_flag_rate": len(mask_set) / max(len(table), 1),
    }

    table.to_csv(os.path.join(args.output_dir, "gap_diagnostic_scores.csv"), index=False)

    # ------------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].scatter(features["peer_valid_mean"], table["instrument_nll"], s=3,
                    color="tab:blue", linewidths=0)
    axes[0].set_xlabel("mean peer valid fraction")
    axes[0].set_ylabel("instrument NLL")
    axes[0].set_title(f"global Spearman {global_corr['peer_valid_mean']:+.3f}, "
                      f"within-chip median {within_median['peer_valid_mean']:+.3f}",
                      fontsize=9)

    axes[1].scatter(table["mask_only_percentile"], table["instrument_percentile"], s=3,
                    color="0.5", linewidths=0)
    axes[1].axhline(THRESHOLD, color="tab:red", lw=0.8, ls="--")
    axes[1].axvline(THRESHOLD, color="tab:red", lw=0.8, ls="--")
    axes[1].set_xlabel("mask-only percentile")
    axes[1].set_ylabel("instrument percentile")
    axes[1].set_title(f"mask-only control: Spearman "
                      f"{agreement['spearman_real_vs_mask_only']:+.3f}, "
                      f"recall {agreement['recall_of_real_by_mask_only']:.2f}", fontsize=9)

    per_chip = table.groupby("chip").apply(
        lambda g: pd.Series({
            "instrument_rate": float((g["instrument_percentile"] >= THRESHOLD).mean()),
            "valid_fraction": float(g["valid_cadences"].mean() / patch.curve_length)}),
        include_groups=False)
    axes[2].scatter(per_chip["valid_fraction"], per_chip["instrument_rate"], s=18,
                    color="tab:purple")
    axes[2].axhline(0.05, color="0.3", lw=0.8, ls="--")
    axes[2].set_xlabel("chip mean valid fraction")
    axes[2].set_ylabel("chip instrument candidate rate")
    chip_rho = float(spearmanr(per_chip["valid_fraction"], per_chip["instrument_rate"])[0])
    axes[2].set_title(f"per chip: Spearman {chip_rho:+.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "gap_diagnostic.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    if surrogate_spread < 1e-4 or agreement["mask_only_flag_rate"] > 0.5:
        verdict = ("control uninformative: peer masks are nearly identical across "
                   "these anchors, so the surrogate carries no signal to compare")
    elif agreement["recall_of_real_by_mask_only"] > 0.5:
        verdict = "gaps dominate"
    elif agreement["recall_of_real_by_mask_only"] > 0.2:
        verdict = "gaps contribute"
    else:
        verdict = "gaps are not the driver"
    summary = {
        "global_spearman_nll_vs_gap_features": global_corr,
        "within_chip_median_spearman": within_median,
        "ridge_r2_gap_features_predicting_nll_heldout": ridge_r2,
        "mask_only_control": agreement,
        "per_chip_spearman_validfraction_vs_candidate_rate": chip_rho,
        "verdict": verdict,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print("\n=== gap features vs instrument NLL (Spearman) ===")
    print(f"  {'feature':24s} {'global':>8s} {'within-chip median':>20s}")
    for column in features.columns:
        print(f"  {column:24s} {global_corr[column]:+8.3f} {within_median[column]:+20.3f}")
    print(f"\nridge R^2, gap features -> instrument NLL (held out): {ridge_r2:+.3f}")
    print(f"per-chip Spearman(valid fraction, candidate rate): {chip_rho:+.3f}")
    print("\n=== mask-only control ===")
    for key, value in agreement.items():
        print(f"  {key:32s} {value}")
    print(f"\nVERDICT: {verdict}")
    print(f"outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
