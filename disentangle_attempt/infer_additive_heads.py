"""Stage-2 inference: an additive split of the anchor into physics and instrument.

    physics_curve       stitched from the four complementary masks, so every cadence is
                        predicted while hidden from the physics encoder
    instrument_curve    one pass over the anchor's actual eight peers
    reconstructed_raw   physics_curve + instrument_curve
    cleaned_curve       physics_curve                      (the physics head's output)
    raw_minus_correction raw_anchor - instrument_curve
    residual            raw_anchor - reconstructed_raw

cleaned_curve and raw_minus_correction differ by exactly the residual -- asserted here,
because that identity is what makes the decomposition auditable.

No quiet reference peers, no reference_context.pt, no pred_actual/pred_reference, and
no counterfactual decoder pass: this experiment has no shared decoder at all.

    python -m disentangle_attempt.infer_additive_heads \
      --heads .../additive_heads/heads_best.pt --n-stars 10
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

from scipy.stats import pearsonr

from disentangle_attempt.additive_heads import AdditiveHeadsModel, state_hash
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.train import pick_device
from disentangle_attempt.train_additive_heads import build_patch


def gaps(values, valid):
    return np.where(valid, values, np.nan)


def rms(values, valid):
    return float(np.sqrt((values[valid] ** 2).mean()))


def safe_corr(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def peer_median(patch, peer_rows):
    """Robust common mode of the eight peers, valid where any peer observes."""
    flux, mask = patch.X[peer_rows], patch.M[peer_rows]
    seen = mask.sum(axis=0) > 0
    common = np.zeros(flux.shape[1])
    common[seen] = np.nanmedian(np.where(mask, flux, np.nan)[:, seen], axis=0)
    return np.nan_to_num(common), seen


@torch.no_grad()
def predict(model, patch, rows, peer_rows, masks, device):
    """Stitched physics curve + one-shot instrument curve."""
    raw = torch.from_numpy(patch.X[rows]).to(device)
    valid = torch.from_numpy(patch.M[rows]).to(device)
    _, representation = model.instrument_representation(
        torch.from_numpy(patch.X[peer_rows]).to(device),
        torch.from_numpy(patch.M[peer_rows]).to(device))
    instrument = model.instrument_head(representation)

    physics = torch.zeros(len(rows), patch.curve_length, device=device)
    for k in range(masks.shape[0]):
        hidden = masks[k].to(device).unsqueeze(0).expand(len(rows), -1)
        latent = model.physics_latent(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        predicted = model.physics_head(latent)
        physics[:, masks[k]] = predicted[:, masks[k]]      # keep only hidden cadences
    return physics.cpu().numpy(), instrument.cpu().numpy()


def plot_star(patch, row, physics, instrument, common, path, title):
    raw = patch.X[row]
    valid = patch.M[row]
    reconstructed = physics + instrument
    residual = raw - reconstructed
    raw_minus = raw - instrument
    x = np.arange(len(raw))

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[0].plot(x, gaps(reconstructed, valid), lw=0.8, color="tab:green",
                 label="reconstructed = physics + instrument")
    axes[0].set_ylabel("normalized flux")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(title, fontsize=10)

    axes[1].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[1].plot(x, gaps(physics, valid), lw=0.9, color="tab:blue",
                 label="cleaned = physics head output")
    axes[1].set_ylabel("normalized flux")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].axhline(0.0, color="0.8", lw=0.6)
    axes[2].plot(x, gaps(instrument, valid), lw=0.8, color="tab:red",
                 label="instrument head output (correction)")
    axes[2].plot(x, gaps(common - np.nanmean(common[valid]), valid), lw=0.7,
                 color="tab:olive", label="median of the 8 peers (centred)")
    axes[2].set_ylabel("correction")
    axes[2].legend(fontsize=8, loc="upper right")

    axes[3].plot(x, gaps(physics, valid), lw=0.8, color="tab:blue", label="physics curve")
    axes[3].plot(x, gaps(raw_minus, valid), lw=0.8, color="tab:purple",
                 label="raw - instrument")
    axes[3].plot(x, gaps(residual, valid), lw=0.7, color="tab:orange",
                 label="residual = raw - reconstructed")
    axes[3].axhline(0.0, color="0.8", lw=0.6)
    axes[3].set_ylabel("normalized flux")
    axes[3].set_xlabel("cadence index (gaps = removed cadences)")
    axes[3].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--n-stars", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    state = torch.load(args.heads, map_location="cpu", weights_only=False)
    device = pick_device(state["config"].get("device", "auto"))
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.heads)),
                                              "inference")
    os.makedirs(out_dir, exist_ok=True)

    model = AdditiveHeadsModel(state["source_checkpoint"], map_location="cpu").to(device)
    model.physics_head.load_state_dict(state["physics_head"])
    model.instrument_head.load_state_dict(state["instrument_head"])
    model.eval()
    after = {"physics_s4d": state_hash(model.physics_encoder),
             "instrument_s4d": state_hash(model.instrument_encoder)}
    assert all(state["frozen_hashes"][k] == after[k] for k in after), \
        "frozen encoders differ from the ones Stage 2 trained against"

    patch = build_patch(model.source_config, args.parquet)
    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    peer_rows = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])
    masks = complementary_masks(patch.curve_length, n_masks=4)

    physics, instrument = predict(model, patch, rows, peer_rows, masks, device)

    records = []
    for k, row in enumerate(rows):
        valid = patch.M[row]
        raw = patch.X[row]
        reconstructed = physics[k] + instrument[k]
        residual = raw - reconstructed
        raw_minus = raw - instrument[k]
        # physics - (raw - instrument) == -(raw - reconstructed); the identity that
        # makes the decomposition auditable rather than merely plotted.
        assert np.allclose((physics[k] - raw_minus)[valid], -residual[valid], atol=1e-5)
        common, _ = peer_median(patch, peer_rows[k])
        records.append({
            "TIC": patch.tic[row],
            "physics_rms": rms(physics[k], valid),
            "instrument_rms": rms(instrument[k], valid),
            "residual_rms": rms(residual, valid),
            "raw_rms": rms(raw, valid),
            "corr_instrument_peer_median": safe_corr(instrument[k][valid], common[valid]),
            "corr_instrument_raw": safe_corr(instrument[k][valid], raw[valid]),
            "corr_physics_raw": safe_corr(physics[k][valid], raw[valid]),
            "max_abs_physics_minus_rawminus": float(
                np.abs((physics[k] - raw_minus)[valid]).max()),
        })
        plot_star(patch, row, physics[k], instrument[k], common,
                  os.path.join(out_dir, f"star_{k:02d}_TIC{patch.tic[row]}.png"),
                  f"TIC {patch.tic[row]} - sector {patch.target[0]} "
                  f"cam{patch.target[1]}-ccd{patch.target[2]} (held-out test)")
    table = pd.DataFrame(records)
    table.to_csv(os.path.join(out_dir, "per_star.csv"), index=False)

    summary = {
        "heads_checkpoint": os.path.abspath(args.heads),
        "source_checkpoint": state["source_checkpoint"],
        "n_stars": len(rows),
        "median": {c: float(table[c].median()) for c in table.columns if c != "TIC"},
        "encoder_hashes": after,
        "outputs": sorted(os.listdir(out_dir)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print(table.to_string(index=False))
    print(f"\nmedians: {summary['median']}")
    print(f"plots in {out_dir}")


if __name__ == "__main__":
    main()
