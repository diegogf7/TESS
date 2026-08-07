"""Regenerate the counterfactual plots with the corrected equations, and audit them.

Every panel here is drawn from the two decoder passes directly, so the arithmetic is
visible rather than asserted:

    correction = pred_actual - pred_reference
    cleaned    = raw_anchor - correction

pred_reference is never labelled "cleaned". It carries the decoder's reconstruction
error, so raw - pred_reference would attribute that error to the instrument -- the bug
this replaces, whose size is reported per star.

Inference only: the checkpoint, its saved reference context, preprocessing, peers and
masks are all used as-is.

    python -m disentangle_attempt.plot_corrected_inference \
      --checkpoint disentangle_attempt/outputs/fast_strict/best.pt
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

from disentangle_attempt.dataset import (CrossSectorPatch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.infer import dual_context_prediction, identity_correction_check
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import build_model
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device


def gaps(values, valid):
    """Removed cadences stay gaps; they are never drawn as zero flux."""
    return np.where(valid, values, np.nan)


def rms(values, valid):
    return float(np.sqrt((values[valid] ** 2).mean()))


def plot_example(cadence_ids, raw, actual, reference, correction, cleaned, valid,
                 title, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    x = np.arange(len(raw))
    axes[0].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[0].plot(x, gaps(actual, valid), lw=0.8, color="tab:green",
                 label="pred_actual (decoded with the actual peers)")
    axes[0].plot(x, gaps(reference, valid), lw=0.8, color="tab:orange",
                 label="pred_reference (decoded with the quiet reference peers)")
    axes[0].set_ylabel("normalized flux")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(title, fontsize=10)

    axes[1].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[1].plot(x, gaps(cleaned, valid), lw=0.9, color="tab:blue",
                 label="cleaned = raw - correction")
    axes[1].set_ylabel("normalized flux")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].axhline(0.0, color="0.7", lw=0.6)
    axes[2].plot(x, gaps(correction, valid), lw=0.8, color="tab:red",
                 label="correction = pred_actual - pred_reference")
    axes[2].set_ylabel("correction")
    axes[2].set_xlabel(f"cadence index (absolute cadence {cadence_ids[0]}-{cadence_ids[-1]};"
                       " gaps = removed cadences)")
    axes[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_raw_vs_cleaned(patch, rows, cleaned, path):
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.0 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row, curve in zip(axes, rows, cleaned):
        valid = patch.M[row]
        x = np.arange(len(valid))
        ax.scatter(x[valid], patch.X[row][valid], s=2, color="0.6", linewidths=0,
                   label="raw")
        ax.scatter(x[valid], curve[valid], s=2, color="tab:blue", linewidths=0,
                   label="cleaned = raw - correction")
        ax.set_ylabel(f"TIC {patch.tic[row]}", fontsize=7)
    axes[0].legend(fontsize=8, markerscale=4, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    fig.suptitle("held-out test stars: raw vs cleaned "
                 "(cleaned = raw - [pred_actual - pred_reference])", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_components(patch, rows, actual, reference, correction, cleaned, path):
    """Every term of the equation on one row per star, so it can be checked by eye."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.4 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, row) in enumerate(zip(axes, rows)):
        valid = patch.M[row]
        x = np.arange(len(valid))
        ax.plot(x, gaps(patch.X[row], valid), lw=0.7, color="0.55", label="raw")
        ax.plot(x, gaps(actual[k], valid), lw=0.7, color="tab:green", label="pred_actual")
        ax.plot(x, gaps(reference[k], valid), lw=0.7, color="tab:orange",
                label="pred_reference")
        ax.plot(x, gaps(cleaned[k], valid), lw=0.8, color="tab:blue",
                label="cleaned = raw - correction")
        ax.plot(x, gaps(correction[k], valid), lw=0.7, color="tab:red",
                label="correction = pred_actual - pred_reference")
        ax.axhline(0.0, color="0.85", lw=0.5)
        ax.set_ylabel(f"TIC {patch.tic[row]}", fontsize=7)
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    fig.suptitle("prediction components: raw, both decoder passes, their difference, "
                 "and the cleaned curve", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--n-stars", type=int, default=6)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output_dir or os.path.join(run_dir, "corrected_inference_plots")
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    sector, camera, ccd = target_from_checkpoint(state, config)
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        peer_min_distance=config.get("peer_min_distance_px", 12.0),
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(config, args.require_cross_sector),
        verbose=False)

    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    target_sector = patch.target[0]
    quiet = load_reference_context(run_dir, expected_cadence_ids=patch.grids[target_sector])
    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    peer_rows = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])

    identity_max = identity_correction_check(
        model, torch.from_numpy(patch.X[rows]), torch.from_numpy(patch.M[rows]),
        torch.from_numpy(patch.X[peer_rows]), torch.from_numpy(patch.M[peer_rows]),
        masks, device)
    assert identity_max < 1e-5, f"identity check failed: {identity_max:.3e}"
    print(f"identity check (actual peers == reference peers): max |correction| "
          f"{identity_max:.3e}", flush=True)

    actual_pred, reference_pred, _, _, _ = dual_context_prediction(
        model, torch.from_numpy(patch.X[rows]), torch.from_numpy(patch.M[rows]),
        torch.from_numpy(patch.X[peer_rows]), torch.from_numpy(patch.M[peer_rows]),
        quiet["peer_raw"].unsqueeze(0).expand(len(rows), -1, -1),
        quiet["peer_mask"].unsqueeze(0).expand(len(rows), -1, -1), masks, device)
    actual = actual_pred.numpy()
    reference = reference_pred.numpy()
    correction = actual - reference
    raw = patch.X[rows]
    cleaned = raw - correction

    assert np.isfinite(actual).all() and np.isfinite(reference).all(), "non-finite predictions"
    for k, row in enumerate(rows):
        valid = patch.M[row]
        assert np.allclose(correction[k][valid], (actual[k] - reference[k])[valid], atol=1e-6)
        assert np.allclose(cleaned[k][valid], (raw[k] - correction[k])[valid], atol=1e-6)
        assert np.allclose((raw[k] - cleaned[k])[valid], correction[k][valid], atol=1e-6)

    records = []
    for k, row in enumerate(rows):
        valid = patch.M[row]
        records.append({
            "TIC": patch.tic[row],
            "corrected_correction_rms": rms(correction[k], valid),
            "old_buggy_correction_rms": rms(raw[k] - reference[k], valid),
            "decoder_reconstruction_rms": rms(raw[k] - actual[k], valid),
        })
    table = pd.DataFrame(records)
    table["inflation_x"] = (table["old_buggy_correction_rms"]
                            / table["corrected_correction_rms"])
    table.to_csv(os.path.join(out_dir, "correction_rms.csv"), index=False)

    cadence_ids = patch.grids[target_sector]
    plot_example(cadence_ids, raw[0], actual[0], reference[0], correction[0], cleaned[0],
                 patch.M[rows[0]],
                 f"TIC {patch.tic[rows[0]]} - sector {target_sector} "
                 f"cam{patch.target[1]}-ccd{patch.target[2]}: "
                 "correction = pred_actual - pred_reference",
                 os.path.join(out_dir, "example_correction.png"))
    plot_raw_vs_cleaned(patch, rows, cleaned, os.path.join(out_dir, "raw_vs_cleaned.png"))
    plot_components(patch, rows, actual, reference, correction, cleaned,
                    os.path.join(out_dir, "prediction_components.png"))

    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint),
                   "identity_max_abs_correction": identity_max,
                   "stars": records,
                   "equations": {"correction": "pred_actual - pred_reference",
                                 "cleaned": "raw_anchor - correction"}},
                  handle, indent=2, default=float)

    print(table.to_string(index=False))
    print(f"\nplots in {out_dir}")


if __name__ == "__main__":
    main()
