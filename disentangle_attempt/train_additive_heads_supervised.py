"""Two additive heads on frozen encoders, with the instrument head SUPERVISED.

Adds two terms to the reconstruction-only experiment:

    L_instrument  the correction should follow the eight-peer common mode
    L_leakage     the physics curve should NOT correlate with that common mode

    L_total = L_recon + 0.25 * L_instrument + 0.05 * L_leakage

The common mode is built from the peers alone -- each peer's own median removed, then
the median across peers where at least four observe, then centred. The anchor never
enters it, so the target cannot leak the answer.

Training uses the SAME four complementary masks as inference, and scores each mask only
on the cadences it hides, so every valid cadence is scored exactly once per pass.

Frozen: physics S4D, instrument S4D, and the original shared decoder (loaded so its
hash can be checked, never called). Only the two heads train.

    python -m disentangle_attempt.train_additive_heads_supervised \
      --source-checkpoint .../base_model/best.pt
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from disentangle_attempt.additive_heads import AdditiveHeadsModel
from disentangle_attempt.dataset import CrossSectorAnchorDataset, audit_batch
from disentangle_attempt.losses import masked_smooth_l1
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.train import load_config, pick_device, set_seed
from disentangle_attempt.train_additive_heads import build_patch, cache_instrument

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_VALID_PEERS = 4


def peer_common_mode(patch, peer_rows):
    """Common mode of the eight peers, and where it is defined. Anchor never used.

    Each peer's own median is removed first so a bright star cannot dominate the median
    across peers; the result is centred to median zero.
    """
    flux = patch.X[peer_rows].astype(np.float64)          # [P, L]
    mask = patch.M[peer_rows]
    centred = np.where(mask, flux, np.nan)
    per_peer_median = np.nanmedian(centred, axis=1, keepdims=True)
    centred = centred - per_peer_median

    counts = mask.sum(axis=0)
    valid = counts >= MIN_VALID_PEERS
    common = np.zeros(flux.shape[1])
    if valid.any():
        common[valid] = np.nanmedian(centred[:, valid], axis=0)
        common[valid] -= np.median(common[valid])          # centre to median zero
    return np.nan_to_num(common).astype(np.float32), valid


def cache_common_mode(patch, split):
    anchors = patch.split_anchors[split]
    peer_rows = patch.peers[split][0]
    curves = np.zeros((len(anchors), patch.curve_length), dtype=np.float32)
    valids = np.zeros((len(anchors), patch.curve_length), dtype=bool)
    for k in range(len(anchors)):
        curves[k], valids[k] = peer_common_mode(patch, peer_rows[k])
    return torch.from_numpy(curves), torch.from_numpy(valids)


def masked_pearson_squared(a, b, mask, min_points=32, min_std=1e-6):
    """Mean over the batch of squared Pearson r on `mask`; rows with no variance skipped."""
    terms = []
    for k in range(a.shape[0]):
        pick = mask[k]
        if int(pick.sum()) < min_points:
            continue
        x, y = a[k][pick], b[k][pick]
        x = x - x.mean()
        y = y - y.mean()
        sx, sy = torch.sqrt((x * x).sum()), torch.sqrt((y * y).sum())
        if float(sx) < min_std or float(sy) < min_std:
            continue                                       # effectively constant
        terms.append(((x * y).sum() / (sx * sy)) ** 2)
    if not terms:
        return a.sum() * 0.0
    return torch.stack(terms).mean()


def run_split(model, patch, split, cache, common, config, masks, device, indices,
              optimizer=None):
    """One pass over `indices`, looping the four complementary masks per batch."""
    anchors = patch.split_anchors[split]
    common_curve, common_valid = common
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "recon": 0.0, "instrument": 0.0, "leakage": 0.0}
    seen = 0
    for index in indices:
        rows = anchors[index.numpy()]
        raw = torch.from_numpy(patch.X[rows]).to(device)
        valid = torch.from_numpy(patch.M[rows]).to(device)
        representation = cache[index].to(device)
        target = common_curve[index].to(device)
        target_valid = common_valid[index].to(device)

        losses = {k: 0.0 for k in totals}
        step_loss = 0.0
        for k in range(masks.shape[0]):
            hidden = masks[k].to(device).unsqueeze(0).expand(len(rows), -1)
            latent = model.physics_latent(raw.masked_fill(hidden, 0.0), valid & ~hidden)
            outputs = model(latent, representation, hidden)
            physics = outputs["predicted_physics"]
            instrument = outputs["predicted_instrument"]

            score_mask = hidden & valid                    # this mask's cadences only
            recon = masked_smooth_l1(physics + instrument, raw, score_mask)
            instrument_loss = masked_smooth_l1(instrument, target,
                                               target_valid & score_mask)
            leakage = masked_pearson_squared(physics, target, score_mask & target_valid)
            total = (recon + float(config["instrument_weight"]) * instrument_loss
                     + float(config["leakage_weight"]) * leakage)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                               float(config["gradient_clip"]))
                optimizer.step()
            losses["recon"] += float(recon.detach())
            losses["instrument"] += float(instrument_loss.detach())
            losses["leakage"] += float(leakage.detach())
            losses["total"] += float(total.detach())
        for key in totals:
            totals[key] += losses[key] / masks.shape[0] * len(rows)
        seen += len(rows)
    return {k: v / max(seen, 1) for k, v in totals.items()}


def epoch_indices(n, steps, batch, generator):
    return [torch.randint(0, n, (batch,), generator=generator) for _ in range(steps)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--config",
                        default=os.path.join(HERE, "config_additive_heads_supervised.yaml"))
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = pick_device(config.get("device", "auto"))
    source_dir = os.path.dirname(os.path.abspath(args.source_checkpoint))
    out_dir = args.output_dir or os.path.join(os.path.dirname(source_dir),
                                              "additive_heads_supervised")
    os.makedirs(out_dir, exist_ok=True)

    model = AdditiveHeadsModel(args.source_checkpoint, map_location="cpu").to(device)
    before = dict(model.frozen_hashes)
    source_config = model.source_config
    print(f"source checkpoint: {os.path.abspath(args.source_checkpoint)}")
    print(f"source config: sector={source_config.get('sector')} "
          f"camera={source_config.get('camera')} ccd={source_config.get('ccd')} "
          f"peer_min_distance_px={source_config.get('peer_min_distance_px')}")
    assert (source_config.get("sector"), source_config.get("camera"),
            source_config.get("ccd")) == (1, 4, 2), "not the Sector 1 cam4-ccd2 model"
    assert float(source_config.get("peer_min_distance_px", -1)) == 12.0, \
        "source checkpoint was not trained with the 12 px exclusion"
    print(f"frozen hashes before: {before}")
    for module in (model.physics_encoder, model.instrument_encoder, model.frozen_decoder):
        assert not any(p.requires_grad for p in module.parameters())

    patch = build_patch(source_config, args.parquet)
    masks = complementary_masks(patch.curve_length, n_masks=int(config["n_masks"]))
    assert bool((masks.sum(dim=0) == 1).all()), "masks must cover every cadence once"
    from torch.utils.data import DataLoader
    datasets = {n: CrossSectorAnchorDataset(patch, n, seed=config["seed"])
                for n in ("train", "val", "test")}
    audit_batch(patch, next(iter(DataLoader(datasets["train"],
                                            batch_size=config["anchors_per_step"],
                                            shuffle=True, drop_last=True))))

    caches = {n: cache_instrument(model, patch, n, device) for n in ("train", "val", "test")}
    commons = {n: cache_common_mode(patch, n) for n in ("train", "val", "test")}
    print(f"cached instrument {{k: v.shape}} and peer common modes", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    handle = open(os.path.join(out_dir, "history.csv"), "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(["epoch", "train_total", "train_recon", "train_instrument",
                     "train_leakage", "val_total", "val_recon", "val_instrument",
                     "val_leakage", "seconds"])

    best = {"val_total": float("inf"), "epoch": -1}
    since_best, started, stop_reason = 0, time.time(), "max_epochs"
    for epoch in range(int(config["max_epochs"])):
        epoch_start = time.time()
        generator = torch.Generator().manual_seed(config["seed"] * 1000 + epoch)
        train_indices = epoch_indices(len(patch.split_anchors["train"]),
                                      int(config["max_train_steps_per_epoch"]),
                                      int(config["anchors_per_step"]), generator)
        train_metrics = run_split(model, patch, "train", caches["train"], commons["train"],
                                  config, masks, device, train_indices, optimizer)
        val_generator = torch.Generator().manual_seed(config["seed"])
        val_indices = epoch_indices(len(patch.split_anchors["val"]),
                                    int(config["max_val_steps_per_epoch"]),
                                    int(config["anchors_per_step"]), val_generator)
        val_metrics = run_split(model, patch, "val", caches["val"], commons["val"],
                                config, masks, device, val_indices)
        writer.writerow([epoch, train_metrics["total"], train_metrics["recon"],
                         train_metrics["instrument"], train_metrics["leakage"],
                         val_metrics["total"], val_metrics["recon"],
                         val_metrics["instrument"], val_metrics["leakage"],
                         round(time.time() - epoch_start, 2)])
        handle.flush()
        print(f"epoch {epoch:2d}: train total {train_metrics['total']:.4f} "
              f"(rec {train_metrics['recon']:.4f} inst {train_metrics['instrument']:.4f} "
              f"leak {train_metrics['leakage']:.4f}) | val total {val_metrics['total']:.4f} "
              f"(rec {val_metrics['recon']:.4f} inst {val_metrics['instrument']:.4f} "
              f"leak {val_metrics['leakage']:.4f}) | {time.time() - epoch_start:.1f}s",
              flush=True)

        if val_metrics["total"] < best["val_total"]:
            best = {"val_total": val_metrics["total"], "epoch": epoch,
                    "train": train_metrics, "val": val_metrics}
            since_best = 0
            torch.save({"physics_head": model.physics_head.state_dict(),
                        "instrument_head": model.instrument_head.state_dict(),
                        "source_checkpoint": os.path.abspath(args.source_checkpoint),
                        "source_config": source_config, "config": config,
                        "epoch": epoch, "frozen_hashes": before},
                       os.path.join(out_dir, "heads_best.pt"))
        else:
            since_best += 1
            if since_best >= int(config["early_stopping_patience"]):
                stop_reason = "early_stopping"
                print(f"early stopping after epoch {epoch}", flush=True)
                break
        if (time.time() - started) / 60 >= float(config["max_runtime_minutes"]):
            stop_reason = "max_runtime"
            break
    handle.close()

    state = torch.load(os.path.join(out_dir, "heads_best.pt"), weights_only=False)
    model.physics_head.load_state_dict(state["physics_head"])
    model.instrument_head.load_state_dict(state["instrument_head"])
    test_generator = torch.Generator().manual_seed(config["seed"])
    test_indices = epoch_indices(len(patch.split_anchors["test"]),
                                 int(config["max_val_steps_per_epoch"]),
                                 int(config["anchors_per_step"]), test_generator)
    test_metrics = run_split(model, patch, "test", caches["test"], commons["test"],
                             config, masks, device, test_indices)

    unchanged, after = model.frozen_unchanged()
    assert unchanged, f"frozen weights changed: {before} -> {after}"
    summary = {
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "source_config": source_config, "config": config,
        "anchors": {k: int(len(v)) for k, v in patch.split_anchors.items()},
        "best_epoch": best["epoch"], "best_train": best.get("train"),
        "best_val": best.get("val"), "test": test_metrics,
        "stop_reason": stop_reason,
        "frozen_hashes_before": before, "frozen_hashes_after": after,
        "frozen_unchanged": bool(unchanged),
        "runtime_minutes": round((time.time() - started) / 60, 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nbest epoch {best['epoch']} | val total {best['val_total']:.4f}")
    print(f"test: {test_metrics}")
    print(f"frozen unchanged: {unchanged} ({after})")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
