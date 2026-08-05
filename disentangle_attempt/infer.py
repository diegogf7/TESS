"""Counterfactual inference: raw -> cleaned under a quiet observed instrument context.

The physics encoder is fed the anchor TIC's curve from a DIFFERENT sector -- never the
anchor sector -- so no artificial masking is needed or used. That one physics latent is
decoded twice, once with the star's actual nearest peers and once with the quiet
reference peers:

    correction = pred_actual - pred_reference
    cleaned    = raw_anchor - correction

The correction is the decoder's estimate of what the actual detector neighbourhood
adds relative to a quiet one. The cleaned curve is a counterfactual, NOT proven ground
truth and NOT a measured correction.

    python -m disentangle_attempt.infer \
      --checkpoint disentangle_attempt/outputs/<run_name>/best.pt --tic-id <TIC_ID>
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disentangle_attempt.dataset import CrossSectorPatch
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------- prediction
@torch.no_grad()
def dual_context_prediction(model, raw, valid, actual_peers, actual_peer_mask,
                            quiet_peers, quiet_peer_mask, masks, device, stitch=True):
    """Predictions under both instrument contexts, sharing one physics pass per mask.

    raw/valid [B, L]; peers [B, P, L]; masks [K, L]. Returns actual [B, L],
    cleaned [B, L], physics tokens [B, K, T, D] and both peer token sets.

    stitch=True keeps only each mask's hidden block, so every cadence was predicted
    while invisible to the physics encoder.
    """
    if not stitch and masks.shape[0] != 1:
        raise ValueError("stitch=False expects exactly one mask")
    B, L = raw.shape
    raw, valid = raw.to(device), valid.to(device)
    actual_tokens, actual_context = model.encode_peers(actual_peers.to(device),
                                                       actual_peer_mask.to(device))
    quiet_tokens, quiet_context = model.encode_peers(quiet_peers.to(device),
                                                     quiet_peer_mask.to(device))

    actual_prediction = torch.zeros(B, L, device=device)
    cleaned_prediction = torch.zeros(B, L, device=device)
    physics_tokens = []
    for k in range(masks.shape[0]):
        hidden = masks[k].to(device).unsqueeze(0).expand(B, L)
        tokens = model.encode_physics(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        latent = tokens.flatten(1)
        with_actual = model.decoder(torch.cat([latent, actual_context], dim=-1))
        with_quiet = model.decoder(torch.cat([latent, quiet_context], dim=-1))
        if stitch:
            actual_prediction[:, masks[k]] = with_actual[:, masks[k]]
            cleaned_prediction[:, masks[k]] = with_quiet[:, masks[k]]
        else:
            actual_prediction, cleaned_prediction = with_actual, with_quiet
        physics_tokens.append(tokens)
    return (actual_prediction.cpu(), cleaned_prediction.cpu(),
            torch.stack(physics_tokens, dim=1).cpu(), actual_tokens.cpu(), quiet_tokens.cpu())


# ------------------------------------------------------------------------ plots
def plot_correction(cadence_ids, raw, reconstructed, cleaned, correction, valid,
                    title, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    x = np.arange(len(raw))
    axes[0].scatter(x[valid], raw[valid], s=2, color="0.55", linewidths=0, label="raw anchor")
    axes[0].scatter(x[valid], reconstructed[valid], s=2, color="tab:green", linewidths=0,
                    label="reconstructed (actual peers)")
    axes[0].set_ylabel("normalized flux")
    axes[0].legend(loc="upper right", fontsize=8, markerscale=4)
    axes[0].set_title(title, fontsize=10)

    axes[1].scatter(x[valid], raw[valid], s=2, color="0.55", linewidths=0, label="raw anchor")
    axes[1].scatter(x[valid], cleaned[valid], s=2, color="tab:blue", linewidths=0,
                    label="cleaned = raw - correction")
    axes[1].set_ylabel("normalized flux")
    axes[1].legend(loc="upper right", fontsize=8, markerscale=4)

    axes[2].axhline(0.0, color="0.7", lw=0.6)
    axes[2].scatter(x[valid], correction[valid], s=2, color="tab:red", linewidths=0,
                    label="correction = pred_actual - pred_reference")
    axes[2].set_xlabel(f"cadence index (absolute cadence {cadence_ids[0]}-{cadence_ids[-1]})")
    axes[2].set_ylabel("correction")
    axes[2].legend(loc="upper right", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history_csv, branch_json, path):
    """Learning curves + the branch-use bar chart, side by side."""
    import csv as _csv
    with open(history_csv) as handle:
        rows = list(_csv.DictReader(handle))
    epochs = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(epochs, [float(r["train_reconstruction"]) for r in rows], "o-",
                 color="tab:blue", label="train (hidden cadences)")
    axes[0].plot(epochs, [float(r["val_reconstruction"]) for r in rows], "o-",
                 color="tab:red", label="validation (hidden cadences)")
    axes[0].plot(epochs, [float(r["val_visible_reconstruction"]) for r in rows], "o--",
                 color="0.6", lw=1, label="validation (visible cadences)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked smooth-L1 (hidden cadences)")
    axes[0].set_title("reconstruction loss", fontsize=10)
    axes[0].legend(fontsize=8)

    with open(branch_json) as handle:
        branch = json.load(handle)
    labels = {"shuffle_physics": "physics inputs\nshuffled across TICs",
              "random_peers": "nearest peers ->\nrandom same-chip peers",
              }
    names = [n for n in labels if n in branch["conditions"]]
    recon = [branch["conditions"][n]["delta_reconstruction"] for n in names]
    cons = [branch["conditions"][n].get("delta_sector_consistency", 0.0) for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.18, recon, 0.36, color="tab:red", label="Δ reconstruction")
    axes[1].bar(x + 0.18, cons, 0.36, color="tab:purple", label="Δ sector consistency")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[n] for n in names], fontsize=7)
    axes[1].set_ylabel("worse ->")
    axes[1].set_title("branch-use controls (positive = branch is used)", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tic-id", default=None)
    parser.add_argument("--parquet", default=None)
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
    masks = complementary_masks(config["curve_length"], n_masks=4)

    row = (patch.row_for_tic(args.tic_id) if args.tic_id
           else int(patch.split_anchors["test"][0]))
    split = patch.split_of_tic(patch.tic[row])
    peer_rows, peer_distances = patch.peers_for_row(row, split)
    print(f"target TIC {patch.tic[row]} (split {split}), sector {sector} "
          f"cam{patch.target[1]}-ccd{patch.target[2]}", flush=True)

    raw = torch.from_numpy(patch.X[row]).unsqueeze(0)
    valid = torch.from_numpy(patch.M[row]).unsqueeze(0)
    actual, cleaned, physics_tokens, actual_tokens, quiet_tokens = dual_context_prediction(
        model, raw, valid,
        torch.from_numpy(patch.X[peer_rows]).unsqueeze(0),
        torch.from_numpy(patch.M[peer_rows]).unsqueeze(0),
        quiet["peer_raw"].unsqueeze(0), quiet["peer_mask"].unsqueeze(0), masks, device)

    raw_np = patch.X[row]
    cleaned_np = cleaned[0].numpy()
    correction = raw_np - cleaned_np
    output_mask = patch.M[row]
    cadence_ids = patch.grids[sector]

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        tic_id=np.int64(patch.tic_int[row]), sector=np.int64(sector),
        camera=np.int64(patch.target[1]), ccd=np.int64(patch.target[2]),
        cadence_ids=cadence_ids, raw_target=raw_np, target_valid_mask=output_mask,
        actual_context_prediction=actual[0].numpy(),
        cleaned_curve=cleaned_np, correction_curve=correction, output_mask=output_mask,
        physics_tokens=physics_tokens[0].numpy(),
        actual_instrument_tokens=actual_tokens[0].numpy(),
        reference_instrument_tokens=quiet_tokens[0].numpy(),
        actual_peer_tic_ids=patch.tic_int[peer_rows],
        actual_peer_distances=peer_distances,
        quiet_peer_tic_ids=np.asarray([patch.tic_int[int(r)] for r in quiet["peer_rows"]]))

    plot_correction(cadence_ids, raw_np, actual[0].numpy(), cleaned_np, correction,
                    output_mask,
                    f"TIC {patch.tic[row]} - sector {sector} cam{patch.target[1]}"
                    f"-ccd{patch.target[2]}: raw vs quiet-context counterfactual",
                    os.path.join(out_dir, "example_correction.png"))
    print(f"correction RMS {np.sqrt((correction[output_mask] ** 2).mean()):.4f} "
          f"(normalized MAD units); wrote inference_arrays.npz, example_correction.png",
          flush=True)

    history = os.path.join(out_dir, "history.csv")
    branch = os.path.join(out_dir, "branch_use_tests.json")
    if os.path.exists(history) and os.path.exists(branch):
        plot_history(history, branch, os.path.join(out_dir, "training_curves.png"))
        print("wrote training_curves.png", flush=True)


if __name__ == "__main__":
    main()
