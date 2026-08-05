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
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------- prediction
@torch.no_grad()
def dual_context_prediction(model, physics_raw, physics_mask, actual_peers,
                            actual_peer_mask, quiet_peers, quiet_peer_mask, device):
    """One cross-sector physics pass, decoded against both instrument contexts.

    physics_raw/mask [B, L] are the anchor TIC's DIFFERENT-sector curve; peers
    [B, P, L] sit on the anchor sector's cadence grid. Returns pred_actual,
    pred_reference, physics tokens and both peer token sets.
    """
    physics_tokens = model.encode_physics(physics_raw.to(device), physics_mask.to(device))
    latent = physics_tokens.flatten(1)
    actual_tokens, actual_context = model.encode_peers(actual_peers.to(device),
                                                       actual_peer_mask.to(device))
    quiet_tokens, quiet_context = model.encode_peers(quiet_peers.to(device),
                                                     quiet_peer_mask.to(device))
    pred_actual = model.decoder(torch.cat([latent, actual_context], dim=-1))
    pred_reference = model.decoder(torch.cat([latent, quiet_context], dim=-1))
    return (pred_actual.cpu(), pred_reference.cpu(), physics_tokens.cpu(),
            actual_tokens.cpu(), quiet_tokens.cpu())


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
                 color="tab:blue", label="train")
    axes[0].plot(epochs, [float(r["val_reconstruction"]) for r in rows], "o-",
                 color="tab:red", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked L1 (valid cadences)")
    axes[0].set_title("reconstruction loss", fontsize=10)
    axes[0].legend(fontsize=8)

    with open(branch_json) as handle:
        branch = json.load(handle)
    labels = {"shuffle_physics": "cross-sector physics ->\ndifferent TIC",
              "random_peers": "nearest peers ->\nrandom same-chip peers"}
    names = [n for n in labels if n in branch["conditions"]]
    recon = [branch["conditions"][n]["delta_reconstruction"] for n in names]
    x = np.arange(len(names))
    axes[1].bar(x, recon, 0.5, color="tab:red", label="Δ reconstruction")
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

    row = (patch.row_for_tic(args.tic_id) if args.tic_id
           else int(patch.split_anchors["test"][0]))
    split = patch.split_of_tic(patch.tic[row])
    peer_rows, peer_distances = patch.peers_for_row(row, split)
    others = patch.other_sector_rows(row)
    assert others, "inference needs the same TIC in a different sector"
    other = int(others[0])
    assert patch.tic[other] == patch.tic[row] and patch.sector[other] != patch.sector[row]
    print(f"target TIC {patch.tic[row]} (split {split}), anchor sector {sector} "
          f"cam{patch.target[1]}-ccd{patch.target[2]}; physics from sector "
          f"{patch.sector[other]}", flush=True)

    pred_actual, pred_reference, physics_tokens, actual_tokens, quiet_tokens = (
        dual_context_prediction(
            model,
            torch.from_numpy(patch.X[other]).unsqueeze(0),
            torch.from_numpy(patch.M[other]).unsqueeze(0),
            torch.from_numpy(patch.X[peer_rows]).unsqueeze(0),
            torch.from_numpy(patch.M[peer_rows]).unsqueeze(0),
            quiet["peer_raw"].unsqueeze(0), quiet["peer_mask"].unsqueeze(0), device))

    raw_np = patch.X[row]
    correction = (pred_actual - pred_reference)[0].numpy()
    cleaned = raw_np - correction
    output_mask = patch.M[row]
    cadence_ids = patch.grids[sector]

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        tic_id=np.int64(patch.tic_int[row]), sector=np.int64(sector),
        physics_sector=np.int64(patch.sector[other]),
        camera=np.int64(patch.target[1]), ccd=np.int64(patch.target[2]),
        cadence_ids=cadence_ids, raw_target=raw_np, target_valid_mask=output_mask,
        actual_context_prediction=pred_actual[0].numpy(),
        reference_context_prediction=pred_reference[0].numpy(),
        correction_curve=correction, cleaned_curve=cleaned, output_mask=output_mask,
        physics_tokens=physics_tokens[0].numpy(),
        actual_instrument_tokens=actual_tokens[0].numpy(),
        reference_instrument_tokens=quiet_tokens[0].numpy(),
        actual_peer_tic_ids=patch.tic_int[peer_rows],
        actual_peer_distances=peer_distances,
        quiet_peer_tic_ids=np.asarray([patch.tic_int[int(r)] for r in quiet["peer_rows"]]))

    plot_correction(cadence_ids, raw_np, pred_actual[0].numpy(), cleaned, correction,
                    output_mask,
                    f"TIC {patch.tic[row]} - anchor sector {sector} cam{patch.target[1]}"
                    f"-ccd{patch.target[2]}, physics from sector {patch.sector[other]}",
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
