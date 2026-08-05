"""Simple before/after: raw curve vs cleaned curve, one panel per star.

    python -m disentangle_attempt.plot_curves \
      --checkpoint disentangle_attempt/outputs/<run_name>/best.pt --n-stars 6
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disentangle_attempt.dataset import CrossSectorPatch
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-stars", type=int, default=6)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))

    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=config.get("sector", "auto"), camera=config.get("camera", "auto"),
        ccd=config.get("ccd", "auto"), curve_length=config["curve_length"],
        n_peers=config["n_peers"], min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        verbose=False)

    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    sector = patch.target[0]
    quiet = load_reference_context(out_dir, expected_cadence_ids=patch.grids[sector])

    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    others = [int(patch.other_sector_rows(r)[0]) for r in rows]
    peers = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])
    for anchor, other in zip(rows, others):
        assert patch.tic[anchor] == patch.tic[other]
        assert patch.sector[anchor] != patch.sector[other]

    pred_actual, pred_reference, _, _, _ = dual_context_prediction(
        model,
        torch.from_numpy(patch.X[others]), torch.from_numpy(patch.M[others]),
        torch.from_numpy(patch.X[peers]), torch.from_numpy(patch.M[peers]),
        quiet["peer_raw"].unsqueeze(0).expand(len(rows), -1, -1),
        quiet["peer_mask"].unsqueeze(0).expand(len(rows), -1, -1), device)
    correction = (pred_actual - pred_reference).numpy()

    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.4 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, anchor, other) in enumerate(zip(axes, rows, others)):
        keep, keep_o = patch.M[anchor], patch.M[other]
        x = np.arange(patch.curve_length)
        cleaned = patch.X[anchor] - correction[k]
        ax.scatter(x[keep], patch.X[anchor][keep], s=2.4, color="0.6", linewidths=0,
                   label=f"anchor target (sector {patch.sector[anchor]})")
        ax.scatter(x[keep_o], patch.X[other][keep_o], s=2.4, color="tab:orange",
                   linewidths=0,
                   label=f"physics encoder input (SAME TIC, sector {patch.sector[other]})")
        ax.scatter(x[keep], cleaned[keep], s=2.4, color="tab:blue", linewidths=0,
                   label="cleaned = anchor - correction")
        ax.set_ylabel(f"TIC {patch.tic[anchor]}", fontsize=7)
    axes[0].legend(loc="upper right", fontsize=7, ncol=3, markerscale=4)
    axes[-1].set_xlabel("cadence index within each curve's OWN sector grid "
                        "(the two sectors are different absolute times)")
    fig.suptitle(f"anchor vs cross-sector physics input vs cleaned - sector {sector} "
                 f"cam{patch.target[1]}-ccd{patch.target[2]} (test stars)", fontsize=11)
    fig.tight_layout()
    path = args.out or os.path.join(out_dir, "raw_vs_cleaned.png")
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
