"""Honest inference plots: this architecture predicts a raw anchor, nothing more.

Four panels per star: raw anchor, the masked physics input the encoder actually saw,
the predicted raw anchor, and

    residual = raw_anchor - predicted_raw_anchor

The residual is labelled RECONSTRUCTION RESIDUAL only. It is not an instrument
correction and not a cleaned physics curve: this model emits a single summed
prediction, so nothing here isolates either component.

    python -m disentangle_attempt.plot_inference_examples \
      --checkpoint .../local_s1_c4_ccd2_12px_p128_i32/best.pt --n-stars 6
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disentangle_attempt.dataset import CrossSectorPatch
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import build_model
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device


def gaps(values, valid):
    return np.where(valid, values, np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--n-stars", type=int, default=6)
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output_dir or os.path.join(run_dir, "inference_examples")
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=config["sector"], camera=config["camera"], ccd=config["ccd"],
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        peer_min_distance=config.get("peer_min_distance_px", 12.0),
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        verbose=False)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    peer_rows = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])
    with torch.no_grad():
        raw = torch.from_numpy(patch.X[rows]).to(device)
        valid = torch.from_numpy(patch.M[rows]).to(device)
        hidden = masks[args.mask_index].to(device).unsqueeze(0).expand(len(rows), -1)
        outputs = model(raw.masked_fill(hidden, 0.0), valid & ~hidden,
                        torch.from_numpy(patch.X[peer_rows]).to(device),
                        torch.from_numpy(patch.M[peer_rows]).to(device), hidden)
    predicted = outputs["predicted_raw_anchor"].cpu().numpy()
    hidden_np = masks[args.mask_index].numpy()

    records = []
    for k, row in enumerate(rows):
        v = patch.M[row]
        raw_np = patch.X[row]
        residual = raw_np - predicted[k]
        physics_input = np.where(hidden_np, 0.0, raw_np)
        records.append({"TIC": patch.tic[row],
                        "residual_rms": float(np.sqrt((residual[v] ** 2).mean())),
                        "raw_rms": float(np.sqrt((raw_np[v] ** 2).mean()))})

        x = np.arange(len(raw_np))
        fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
        axes[0].plot(x, gaps(raw_np, v), lw=0.8, color="0.55", label="raw anchor")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_title(f"TIC {patch.tic[row]} - sector {patch.sector[row]} "
                          f"cam{patch.camera[row]}-ccd{patch.ccd[row]} (held-out test)",
                          fontsize=10)
        axes[1].plot(x, gaps(physics_input, v), lw=0.8, color="tab:orange",
                     label=f"masked physics input (mask {args.mask_index}: hidden -> 0)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[2].plot(x, gaps(raw_np, v), lw=0.7, color="0.7", label="raw anchor")
        axes[2].plot(x, gaps(predicted[k], v), lw=0.9, color="tab:green",
                     label="predicted raw anchor")
        axes[2].legend(fontsize=8, loc="upper right")
        axes[3].axhline(0.0, color="0.8", lw=0.6)
        axes[3].plot(x, gaps(residual, v), lw=0.8, color="tab:red",
                     label="reconstruction residual = raw - predicted")
        axes[3].legend(fontsize=8, loc="upper right")
        axes[3].set_xlabel("cadence index (gaps = removed cadences)")
        for ax in axes[:3]:
            ax.set_ylabel("normalized flux")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"star_{k:02d}_TIC{patch.tic[row]}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)

    table = pd.DataFrame(records)
    table.to_csv(os.path.join(out_dir, "residuals.csv"), index=False)
    print(table.to_string(index=False))
    print(f"\nplots in {out_dir}")


if __name__ == "__main__":
    main()
