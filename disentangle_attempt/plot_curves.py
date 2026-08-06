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
from disentangle_attempt.masking import complementary_masks
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
        n_peers=config["n_peers"],
        peer_min_distance=config.get("peer_min_distance_px", 12.0),
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
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
    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    peers = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])
    raw = torch.from_numpy(patch.X[rows])
    valid = torch.from_numpy(patch.M[rows])
    actual_pred, reference_pred, _, _, _ = dual_context_prediction(
        model, raw, valid,
        torch.from_numpy(patch.X[peers]), torch.from_numpy(patch.M[peers]),
        quiet["peer_raw"].unsqueeze(0).expand(len(rows), -1, -1),
        quiet["peer_mask"].unsqueeze(0).expand(len(rows), -1, -1), masks, device)
    # correction = pred_actual - pred_reference; cleaned = raw - correction.
    correction = (actual_pred - reference_pred).numpy()

    hidden = masks[args.mask_index].numpy()
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.2 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, row) in enumerate(zip(axes, rows)):
        curve = patch.X[row] - correction[k]
        keep = patch.M[row]
        x = np.arange(len(keep))
        # Both series exist at EVERY valid cadence; the encoder input is the anchor
        # where visible and normalized zero where hidden -- literally what it is fed.
        encoder_input = np.where(hidden, 0.0, patch.X[row])
        ax.scatter(x[keep], patch.X[row][keep], s=11, color="0.72", linewidths=0,
                   label="anchor / raw target (all valid cadences)")
        ax.scatter(x[keep], encoder_input[keep], s=2.0, color="tab:orange", linewidths=0,
                   label=f"physics encoder input (mask {args.mask_index}: hidden -> 0)")
        ax.scatter(x[keep], curve[keep], s=2.0, color="tab:blue", linewidths=0,
                   label="cleaned = raw - correction")
        ax.set_ylabel(f"TIC {patch.tic[row]}", fontsize=7)
    axes[0].legend(loc="upper right", fontsize=7, ncol=3, markerscale=4)
    axes[-1].set_xlabel("cadence index")
    fig.suptitle(f"raw vs cleaned - sector {sector} cam{patch.target[1]}-ccd{patch.target[2]}"
                 " (test stars)", fontsize=11)
    fig.tight_layout()
    path = args.out or os.path.join(out_dir, "raw_vs_cleaned.png")
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
