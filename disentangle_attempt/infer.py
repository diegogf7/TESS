"""Counterfactual inference: raw -> cleaned under a quiet observed instrument context.

The physics encoder is fed the anchor's OWN curve -- same TIC, same sector, same
cadence grid -- under four complementary masks that tile all 1024 cadences, so every
output cadence is predicted while hidden. Each mask's physics latent is decoded twice,
once with the star's actual nearest same-sector/camera/CCD peers and once with the
quiet reference peers:

    correction = pred_actual - pred_reference
    cleaned    = raw_anchor - correction

pred_reference is NOT the cleaned curve: it carries the decoder's reconstruction error
too, so subtracting it from raw would attribute that error to the instrument. Only the
DIFFERENCE between the two decoder passes isolates what the actual neighbourhood adds
relative to a quiet one.

The cleaned curve is a counterfactual, NOT proven ground truth and NOT a measured
correction.

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
from disentangle_attempt.model import build_model
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------- prediction
@torch.no_grad()
def dual_context_prediction(model, raw, valid, actual_peers, actual_peer_mask,
                            reference_peers, reference_peer_mask, masks, device,
                            stitch=True):
    """Predictions under both instrument contexts, sharing one physics pass per mask.

    Returns the TWO DECODER PREDICTIONS, not a cleaned curve:

        actual_prediction    decoded with the target's real nearest peers
        reference_prediction decoded with the quiet reference peers

    The caller forms
        correction = actual_prediction - reference_prediction
        cleaned    = raw - correction
    Treating reference_prediction as the cleaned curve is wrong: it also carries the
    decoder's reconstruction error and whatever stellar detail the physics latent
    failed to encode, and both would land in the "correction".

    raw/valid [B, L]; peers [B, P, L]; masks [K, L].

    stitch=True keeps only each mask's hidden block, so every cadence was predicted
    while invisible to the physics encoder.
    """
    if not stitch and masks.shape[0] != 1:
        raise ValueError("stitch=False expects exactly one mask")
    B, L = raw.shape
    raw, valid = raw.to(device), valid.to(device)
    actual_tokens, actual_context = model.encode_peers(actual_peers.to(device),
                                                       actual_peer_mask.to(device))
    reference_tokens, reference_context = model.encode_peers(
        reference_peers.to(device), reference_peer_mask.to(device))

    actual_prediction = torch.zeros(B, L, device=device)
    reference_prediction = torch.zeros(B, L, device=device)
    physics_tokens = []
    for k in range(masks.shape[0]):
        hidden = masks[k].to(device).unsqueeze(0).expand(B, L)
        tokens = model.encode_physics(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        latent = model.physics_vector(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        with_actual = model.decoder(torch.cat([latent, actual_context], dim=-1))
        with_reference = model.decoder(torch.cat([latent, reference_context], dim=-1))
        if stitch:
            actual_prediction[:, masks[k]] = with_actual[:, masks[k]]
            reference_prediction[:, masks[k]] = with_reference[:, masks[k]]
        else:
            actual_prediction, reference_prediction = with_actual, with_reference
        physics_tokens.append(tokens)
    return (actual_prediction.cpu(), reference_prediction.cpu(),
            torch.stack(physics_tokens, dim=1).cpu(), actual_tokens.cpu(),
            reference_tokens.cpu())


@torch.no_grad()
def identity_correction_check(model, raw, valid, peers, peer_mask, masks, device):
    """Identical contexts must cancel: correction ~ 0, cleaned ~ raw.

    Anything above numerical noise here means the two decoder passes are not being
    differenced, which is exactly the bug this guards.
    """
    actual, reference, _, _, _ = dual_context_prediction(
        model, raw, valid, peers, peer_mask, peers, peer_mask, masks, device)
    correction = (actual - reference).numpy()
    return float(np.abs(correction[valid.numpy()]).max())


# ------------------------------------------------------------------------ plots
def plot_correction(cadence_ids, raw, reconstructed, cleaned, correction, valid,
                    title, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    x = np.arange(len(raw))
    axes[0].scatter(x[valid], raw[valid], s=2, color="0.55", linewidths=0, label="raw anchor")
    axes[0].scatter(x[valid], reconstructed[valid], s=2, color="tab:green", linewidths=0,
                    label="pred_actual (actual peers)")
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
        n_peers=config["n_peers"],
        peer_min_distance=config.get("peer_min_distance_px", 12.0),
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        verbose=False)

    model = build_model(config).to(device)
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
    actual_pred, reference_pred, physics_tokens, actual_tokens, reference_tokens = dual_context_prediction(
        model, raw, valid,
        torch.from_numpy(patch.X[peer_rows]).unsqueeze(0),
        torch.from_numpy(patch.M[peer_rows]).unsqueeze(0),
        quiet["peer_raw"].unsqueeze(0), quiet["peer_mask"].unsqueeze(0), masks, device)

    raw_np = patch.X[row]
    actual_np = actual_pred[0].numpy()
    reference_np = reference_pred[0].numpy()
    # The correction is the DIFFERENCE BETWEEN THE TWO DECODER PASSES -- what the actual
    # neighbourhood adds relative to a quiet one. Using raw - reference instead would
    # fold the decoder's reconstruction error into the correction.
    correction = actual_np - reference_np
    cleaned_np = raw_np - correction
    output_mask = patch.M[row]
    cadence_ids = patch.grids[sector]

    valid = patch.M[row]
    assert np.allclose(correction[valid], (actual_np - reference_np)[valid], atol=1e-6)
    assert np.allclose(cleaned_np[valid], (raw_np - correction)[valid], atol=1e-6)

    identity_max = identity_correction_check(
        model, torch.from_numpy(patch.X[row]).unsqueeze(0),
        torch.from_numpy(patch.M[row]).unsqueeze(0),
        torch.from_numpy(patch.X[peer_rows]).unsqueeze(0),
        torch.from_numpy(patch.M[peer_rows]).unsqueeze(0), masks, device)
    assert identity_max < 1e-5, \
        f"identical contexts gave a correction of {identity_max:.3e}, expected ~0"
    print(f"identity check (actual peers == reference peers): max |correction| "
          f"{identity_max:.3e}", flush=True)

    np.savez(
        os.path.join(out_dir, "inference_arrays.npz"),
        tic_id=np.int64(patch.tic_int[row]), sector=np.int64(sector),
        camera=np.int64(patch.target[1]), ccd=np.int64(patch.target[2]),
        cadence_ids=cadence_ids, raw_target=raw_np, target_valid_mask=output_mask,
        actual_context_prediction=actual_np,
        reference_context_prediction=reference_np,
        correction_curve=correction, cleaned_curve=cleaned_np, output_mask=output_mask,
        physics_tokens=physics_tokens[0].numpy(),
        actual_instrument_tokens=actual_tokens[0].numpy(),
        reference_instrument_tokens=reference_tokens[0].numpy(),
        actual_peer_tic_ids=patch.tic_int[peer_rows],
        actual_peer_distances=peer_distances,
        quiet_peer_tic_ids=np.asarray([patch.tic_int[int(r)] for r in quiet["peer_rows"]]))

    plot_correction(cadence_ids, raw_np, actual_np, cleaned_np, correction,
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
