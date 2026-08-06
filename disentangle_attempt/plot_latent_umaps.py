"""UMAP views of the two latent spaces of a trained masked same-sector checkpoint.

Read-only: no training, no decoder, no cleaning, no anomaly scoring. UMAP here is a
visualization, not a detector -- distance in these plots is not evidence of anything.

Physics latent    the anchor encoded under every complementary training mask, the four
                  [32, 16] -> [512] token sets averaged into one stable vector. The
                  encoder only ever saw masked inputs, so an unmasked pass would be
                  off-distribution.
Instrument latent the anchor's 8 nearest same-chip peers, each encoded separately by
                  the shared instrument encoder, then averaged.

The two spaces are fit with SEPARATE UMAP models: coordinates from independently
learned embeddings are not comparable, so a joint fit would invent structure.

    ~/tess-venv/bin/python -m disentangle_attempt.plot_latent_umaps \
      --checkpoint disentangle_attempt/outputs/fast_strict/best.pt \
      --max-samples 1000 \
      --output-dir disentangle_attempt/outputs/fast_strict/umaps
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from disentangle_attempt.dataset import (CrossSectorPatch,
                                        infer_require_cross_sector)
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

UMAP_KWARGS = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=42)


@torch.no_grad()
def physics_latents(model, patch, rows, masks, device, batch=64):
    """[N, 512]: mean over the complementary masks of the flattened physics tokens."""
    out = np.zeros((len(rows), model.latent_size), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        raw = torch.from_numpy(patch.X[chunk]).to(device)
        valid = torch.from_numpy(patch.M[chunk]).to(device)
        stacked = []
        for k in range(masks.shape[0]):
            hidden = masks[k].to(device).unsqueeze(0).expand(len(chunk), -1)
            tokens = model.encode_physics(raw.masked_fill(hidden, 0.0), valid & ~hidden)
            stacked.append(tokens.flatten(1))
        out[start:start + len(chunk)] = torch.stack(stacked).mean(0).cpu().numpy()
    return out


@torch.no_grad()
def instrument_latents(model, patch, peer_rows, device, batch=32):
    """[N, 512]: each of the 8 peers encoded separately, then averaged."""
    out = np.zeros((len(peer_rows), model.latent_size), dtype=np.float32)
    for start in range(0, len(peer_rows), batch):
        chunk = peer_rows[start:start + batch]
        flux = torch.from_numpy(patch.X[chunk]).to(device)
        mask = torch.from_numpy(patch.M[chunk]).to(device)
        tokens, _ = model.encode_peers(flux, mask)          # [B, 8, 32, 16]
        out[start:start + len(chunk)] = tokens.flatten(2).mean(1).cpu().numpy()
    return out


def fit_umap(latents, label):
    import umap
    scaled = StandardScaler().fit_transform(latents)
    embedding = umap.UMAP(**UMAP_KWARGS).fit_transform(scaled)
    print(f"  {label}: {latents.shape} -> {embedding.shape}", flush=True)
    return np.asarray(embedding)


def scatter(ax, coords, values, title, cmap="viridis", vmin=None, vmax=None):
    art = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=5, cmap=cmap, vmin=vmin,
                     vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return art


def comparison_plot(physics, instrument, values, label, path, cmap="viridis",
                    categorical=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    if categorical is None:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        art = scatter(axes[0], physics, values, "physics latent", cmap, vmin, vmax)
        scatter(axes[1], instrument, values, "instrument latent", cmap, vmin, vmax)
        fig.colorbar(art, ax=axes, fraction=0.03, label=label)
    else:
        for name, colour in categorical.items():
            pick = values == name
            axes[0].scatter(physics[pick, 0], physics[pick, 1], s=5, color=colour,
                            label=name, linewidths=0)
            axes[1].scatter(instrument[pick, 0], instrument[pick, 1], s=5, color=colour,
                            label=name, linewidths=0)
        for ax, title in zip(axes, ("physics latent", "instrument latent")):
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f"separate UMAPs of the two latent spaces, coloured by {label} "
                 "(visualization only -- distance is not an anomaly score)", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    started = time.time()
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output_dir or os.path.join(run_dir, "umaps")
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    target = state.get("target")
    sector, camera, ccd = target if target else ("auto", "auto", "auto")

    # Same split rule this checkpoint trained under, so `split` labels are truthful.
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(
            config, args.require_cross_sector), verbose=False)

    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    assert not model.training, "model must be in eval mode"

    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows, splits, peer_rows = [], [], []
    for split in ("train", "val", "test"):
        for anchor in patch.split_anchors[split]:
            rows.append(int(anchor))
            splits.append(split)
            peer_rows.append(patch.peers_for_row(int(anchor), split)[0])
    rows = np.asarray(rows)
    peer_rows = np.stack(peer_rows)
    if len(rows) > args.max_samples:                  # deterministic subsample
        pick = np.linspace(0, len(rows) - 1, args.max_samples).astype(int)
        rows, peer_rows = rows[pick], peer_rows[pick]
        splits = [splits[i] for i in pick]
    assert len(np.unique(rows)) == len(rows), "anchors must be unique"
    print(f"{len(rows)} unique anchors on sector {patch.target[0]} "
          f"cam{patch.target[1]}-ccd{patch.target[2]}", flush=True)

    # Strict flag policy holds for every curve that reaches the encoders.
    for name, check in (("anchors", rows), ("peers", peer_rows.reshape(-1))):
        assert not (patch.M[check] & (patch.F[check] != 0)).any(), f"{name}: TESS flag visible"
        assert not (patch.M[check] & (patch.G[check] != 0)).any(), f"{name}: TGLC flag visible"

    physics = physics_latents(model, patch, rows, masks, device)
    instrument = instrument_latents(model, patch, peer_rows, device)
    assert physics.shape == (len(rows), 512), physics.shape
    assert instrument.shape == (len(rows), 512), instrument.shape
    assert np.isfinite(physics).all() and np.isfinite(instrument).all(), "non-finite latents"
    np.save(os.path.join(out_dir, "physics_latents.npy"), physics)
    np.save(os.path.join(out_dir, "instrument_latents.npy"), instrument)

    print("fitting two separate UMAPs", flush=True)
    physics_xy = fit_umap(physics, "physics")
    instrument_xy = fit_umap(instrument, "instrument")

    table = pd.DataFrame({
        "TIC": patch.tic[rows], "split": splits,
        "sector": patch.sector[rows], "camera": patch.camera[rows], "ccd": patch.ccd[rows],
        "detector_x": patch.det_x[rows], "detector_y": patch.det_y[rows],
        "valid_cadences": patch.n_valid[rows],
        "physics_umap_1": physics_xy[:, 0], "physics_umap_2": physics_xy[:, 1],
        "instrument_umap_1": instrument_xy[:, 0], "instrument_umap_2": instrument_xy[:, 1],
    })
    table.to_csv(os.path.join(out_dir, "umap_coordinates.csv"), index=False)

    for coords, name, title in ((physics_xy, "physics_umap.png", "physics latent"),
                                (instrument_xy, "instrument_umap.png", "instrument latent")):
        fig, ax = plt.subplots(figsize=(7, 6))
        art = scatter(ax, coords, table["detector_x"], f"{title} UMAP (colour = detector X)")
        fig.colorbar(art, ax=ax, fraction=0.04, label="detector X (px)")
        fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches="tight")
        plt.close(fig)

    comparison_plot(physics_xy, instrument_xy, table["detector_x"].to_numpy(),
                    "detector X (px)", os.path.join(out_dir, "umap_comparison.png"))
    comparison_plot(physics_xy, instrument_xy, table["detector_y"].to_numpy(),
                    "detector Y (px)", os.path.join(out_dir, "umap_comparison_detector_y.png"),
                    cmap="plasma")
    comparison_plot(physics_xy, instrument_xy, table["split"].to_numpy(), "split",
                    os.path.join(out_dir, "umap_comparison_split.png"),
                    categorical={"train": "0.6", "val": "tab:orange", "test": "tab:blue"})

    print(f"\nexamples: {len(rows)}")
    print(f"physics latents: {physics.shape}   instrument latents: {instrument.shape}")
    print("files:")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {os.path.join(out_dir, name)}")
    print(f"runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
