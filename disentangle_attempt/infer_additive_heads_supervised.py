"""Inference for the supervised additive heads, with cleaned = raw - correction.

    physics_curve        stitched from the four complementary masks
    instrument_correction one pass over the anchor's actual eight peers
    cleaned_curve        raw_anchor - instrument_correction      <- the primary output
    reconstructed_raw    physics_curve + instrument_correction
    residual             raw_anchor - reconstructed_raw

The physics-head output is NOT labelled the cleaned curve here. It differs from
raw - correction by exactly the residual, which is reported rather than hidden.

No quiet reference peers, no reference_context.pt, no counterfactual decoder pass, and
the original shared decoder is never called.

    python -m disentangle_attempt.infer_additive_heads_supervised \
      --heads .../additive_heads_supervised/heads_best.pt \
      --compare-heads .../additive_heads/heads_best.pt --n-stars 10
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

from disentangle_attempt.additive_heads import AdditiveHeadsModel
from disentangle_attempt.losses import masked_smooth_l1
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.train import pick_device
from disentangle_attempt.train_additive_heads import build_patch
from disentangle_attempt.train_additive_heads_supervised import peer_common_mode


def gaps(values, valid):
    return np.where(valid, values, np.nan)


def rms(values, valid):
    return float(np.sqrt((values[valid] ** 2).mean()))


def safe_corr(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


@torch.no_grad()
def predict(model, patch, rows, peer_rows, masks, device):
    raw = torch.from_numpy(patch.X[rows]).to(device)
    valid = torch.from_numpy(patch.M[rows]).to(device)
    _, representation = model.instrument_representation(
        torch.from_numpy(patch.X[peer_rows]).to(device),
        torch.from_numpy(patch.M[peer_rows]).to(device))
    instrument = model.instrument_head(representation)

    physics = torch.zeros(len(rows), patch.curve_length, device=device)
    covered = torch.zeros(patch.curve_length, dtype=torch.long)
    for k in range(masks.shape[0]):
        hidden = masks[k].to(device).unsqueeze(0).expand(len(rows), -1)
        latent = model.physics_latent(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        physics[:, masks[k]] = model.physics_head(latent)[:, masks[k]]
        covered += masks[k].long()
    assert bool((covered == 1).all()), "each cadence must be predicted by exactly one mask"
    return physics.cpu().numpy(), instrument.cpu().numpy()


def evaluate(model, patch, rows, peer_rows, masks, device, label):
    """Metrics shared by both experiments, so the comparison is like-for-like."""
    physics, instrument = predict(model, patch, rows, peer_rows, masks, device)
    records = []
    for k, row in enumerate(rows):
        valid = patch.M[row]
        raw = patch.X[row]
        cleaned = raw - instrument[k]
        reconstructed = physics[k] + instrument[k]
        residual = raw - reconstructed
        common, common_valid = peer_common_mode(patch, peer_rows[k])
        both = valid & common_valid
        records.append({
            "experiment": label, "TIC": patch.tic[row],
            "recon_smooth_l1": float(masked_smooth_l1(
                torch.from_numpy(reconstructed), torch.from_numpy(raw),
                torch.from_numpy(valid))),
            "residual_rms": rms(residual, valid),
            "corr_correction_common": safe_corr(instrument[k][both], common[both]),
            "corr_physics_common": safe_corr(physics[k][both], common[both]),
            "rms_physics_minus_cleaned": rms(physics[k] - cleaned, valid),
            "physics_rms": rms(physics[k], valid),
            "instrument_rms": rms(instrument[k], valid),
        })
    return physics, instrument, pd.DataFrame(records)


def plot_star(patch, row, physics, instrument, common, common_valid, path, title):
    raw = patch.X[row]
    valid = patch.M[row]
    cleaned = raw - instrument
    reconstructed = physics + instrument
    residual = raw - reconstructed
    x = np.arange(len(raw))

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[0].plot(x, gaps(reconstructed, valid), lw=0.8, color="tab:green",
                 label="reconstructed = physics + correction")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(title, fontsize=10)

    axes[1].plot(x, gaps(raw, valid), lw=0.8, color="0.55", label="raw anchor")
    axes[1].plot(x, gaps(cleaned, valid), lw=0.9, color="tab:blue",
                 label="cleaned = raw - correction")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].axhline(0.0, color="0.8", lw=0.6)
    axes[2].plot(x, gaps(instrument, valid), lw=0.8, color="tab:red",
                 label="instrument correction")
    axes[2].plot(x, gaps(common, valid & common_valid), lw=0.7, color="tab:olive",
                 label="peer common mode (centred)")
    axes[2].legend(fontsize=8, loc="upper right")

    axes[3].plot(x, gaps(physics, valid), lw=0.8, color="tab:purple",
                 label="physics-head curve")
    axes[3].plot(x, gaps(cleaned, valid), lw=0.8, color="tab:blue",
                 label="raw - correction")
    axes[3].legend(fontsize=8, loc="upper right")

    axes[4].axhline(0.0, color="0.8", lw=0.6)
    axes[4].plot(x, gaps(residual, valid), lw=0.8, color="tab:orange",
                 label="residual = raw - reconstructed")
    axes[4].set_xlabel("cadence index (gaps = removed cadences)")
    axes[4].legend(fontsize=8, loc="upper right")
    for ax in axes[:4]:
        ax.set_ylabel("normalized flux")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = AdditiveHeadsModel(state["source_checkpoint"], map_location="cpu").to(device)
    model.physics_head.load_state_dict(state["physics_head"])
    model.instrument_head.load_state_dict(state["instrument_head"])
    model.eval()
    unchanged, current = model.frozen_unchanged()
    assert unchanged, "frozen weights differ from training time"
    return model, state, current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", required=True)
    parser.add_argument("--compare-heads", default=None)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--n-stars", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = pick_device("auto")
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.heads)),
                                              "inference")
    os.makedirs(out_dir, exist_ok=True)

    model, state, hashes = load_model(args.heads, device)
    patch = build_patch(model.source_config, args.parquet)
    rows = [int(r) for r in patch.split_anchors["test"][:args.n_stars]]
    peer_rows = np.stack([patch.peers_for_row(r, "test")[0] for r in rows])
    peer_distances = np.stack([patch.peers_for_row(r, "test")[1] for r in rows])
    masks = complementary_masks(patch.curve_length, n_masks=4)

    physics, instrument, table = evaluate(model, patch, rows, peer_rows, masks, device,
                                          "supervised")

    for k, row in enumerate(rows):
        valid = patch.M[row]
        raw = patch.X[row]
        cleaned = raw - instrument[k]
        reconstructed = physics[k] + instrument[k]
        assert np.allclose((cleaned + instrument[k])[valid], raw[valid], atol=1e-5)
        assert np.allclose(reconstructed[valid], (physics[k] + instrument[k])[valid],
                           atol=1e-6)
        assert np.isfinite(physics[k][valid]).all() and np.isfinite(instrument[k][valid]).all()
        common, common_valid = peer_common_mode(patch, peer_rows[k])
        np.savez(os.path.join(out_dir, f"star_{k:02d}_TIC{patch.tic[row]}.npz"),
                 raw_anchor=raw, physics_curve=physics[k], instrument_correction=instrument[k],
                 cleaned_curve=cleaned, reconstructed_raw=reconstructed,
                 reconstruction_residual=raw - reconstructed, peer_common_mode=common,
                 peer_common_valid=common_valid, valid_mask=valid,
                 tic=patch.tic[row], sector=patch.sector[row], camera=patch.camera[row],
                 ccd=patch.ccd[row], peer_tics=patch.tic[peer_rows[k]],
                 peer_distances=peer_distances[k])
        plot_star(patch, row, physics[k], instrument[k], common, common_valid,
                  os.path.join(out_dir, f"star_{k:02d}_TIC{patch.tic[row]}.png"),
                  f"TIC {patch.tic[row]} - sector {patch.sector[row]} "
                  f"cam{patch.camera[row]}-ccd{patch.ccd[row]} (held-out test, supervised)")

    frames = [table]
    if args.compare_heads and os.path.exists(args.compare_heads):
        other, _, _ = load_model(args.compare_heads, device)
        _, _, other_table = evaluate(other, patch, rows, peer_rows, masks, device,
                                     "reconstruction_only")
        frames.append(other_table)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "per_star.csv"), index=False)

    medians = combined.groupby("experiment").median(numeric_only=True)
    verdict = None
    if len(frames) == 2:
        s = medians.loc["supervised"]
        r = medians.loc["reconstruction_only"]
        verdict = {
            "correction_common_correlation_up": bool(
                s["corr_correction_common"] > r["corr_correction_common"]),
            "physics_common_correlation_down": bool(
                abs(s["corr_physics_common"]) < abs(r["corr_physics_common"])),
            "reconstruction_within_10pct": bool(
                s["recon_smooth_l1"] <= 1.10 * r["recon_smooth_l1"]),
        }
        verdict["promising"] = bool(all(verdict.values()))

    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump({"heads": os.path.abspath(args.heads),
                   "source_checkpoint": state["source_checkpoint"],
                   "frozen_hashes": hashes, "n_stars": len(rows),
                   "medians": medians.to_dict(), "verdict": verdict},
                  handle, indent=2, default=float)

    print(medians.to_string())
    if verdict:
        print(f"\nverdict: {verdict}")
    print(f"\noutputs in {out_dir}")


if __name__ == "__main__":
    main()
