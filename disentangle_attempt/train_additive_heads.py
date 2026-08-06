"""Stage 2: train two additive MLP heads on FROZEN Stage-1 encoders.

    predicted_raw = physics_head(physics_latent) + instrument_head(instrument_context)

Only the two heads are optimized. The encoders are eval-mode, requires_grad_(False),
and their weight hashes are compared before and after. The Stage-1 shared decoder is
never loaded.

No loss beyond masked Smooth-L1 on hidden valid cadences. Nothing pushes the heads
towards a physical split, so a collapse -- one head reproducing the anchor while the
other goes near-constant -- is a possible outcome and is measured, not prevented.

Instrument representations are cached once per anchor: peers do not depend on the
random mask, so re-encoding 256 peer curves every step would be wasted work.

    python -m disentangle_attempt.train_additive_heads \
      --source-checkpoint .../base_model/best.pt \
      --config disentangle_attempt/config_additive_heads.yaml
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from disentangle_attempt.additive_heads import AdditiveHeadsModel, state_hash
from disentangle_attempt.dataset import (CrossSectorAnchorDataset, CrossSectorPatch,
                                         audit_batch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.losses import masked_smooth_l1
from disentangle_attempt.masking import mask_views
from disentangle_attempt.train import DEFAULT_PARQUET, load_config, pick_device, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))


def build_patch(source_config, parquet, require_cross_sector="auto"):
    """Dataset geometry comes from the SOURCE checkpoint, so both stages agree."""
    sector, camera, ccd = target_from_checkpoint({"target": None}, source_config)
    return CrossSectorPatch(
        parquet or source_config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=source_config["curve_length"], n_peers=source_config["n_peers"],
        peer_min_distance=source_config.get("peer_min_distance_px", 12.0),
        min_valid_fraction=source_config.get("min_valid_fraction", 0.5),
        split_seed=source_config["seed"],
        max_eligible_anchors=source_config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(source_config, require_cross_sector),
        verbose=True)


@torch.no_grad()
def cache_instrument(model, patch, split, device, batch=32):
    """[n_anchors, 4096] once: the peer group never changes with the mask."""
    anchors = patch.split_anchors[split]
    peer_rows = patch.peers[split][0]
    out = torch.zeros(len(anchors), model.n_peers * model.latent_size)
    for start in range(0, len(anchors), batch):
        rows = peer_rows[start:start + batch]
        _, representation = model.instrument_representation(
            torch.from_numpy(patch.X[rows]).to(device),
            torch.from_numpy(patch.M[rows]).to(device))
        out[start:start + len(rows)] = representation.cpu()
    return out


def epoch_indices(n, steps, batch, generator):
    for _ in range(steps):
        yield torch.randint(0, n, (batch,), generator=generator)


def run_split(model, patch, split, cache, config, device, steps, generator,
              optimizer=None, shuffle=None):
    """One pass. `shuffle` rolls one branch's latents across the batch (a control)."""
    anchors = patch.split_anchors[split]
    training = optimizer is not None
    model.train(training)
    totals, seen = 0.0, 0
    for index in epoch_indices(len(anchors), steps, config["anchors_per_step"], generator):
        rows = anchors[index.numpy()]
        raw = torch.from_numpy(patch.X[rows])
        valid = torch.from_numpy(patch.M[rows])
        masked, hidden, visible = mask_views(raw, valid, config["hidden_fraction"],
                                             generator=generator)
        physics = model.physics_latent(masked.to(device), visible.to(device))
        instrument = cache[index].to(device)
        if shuffle == "physics":
            physics = physics.roll(1, dims=0)
        elif shuffle == "instrument":
            instrument = instrument.roll(1, dims=0)

        outputs = model(physics, instrument, hidden.to(device))
        loss = masked_smooth_l1(outputs["predicted_raw_anchor"], raw.to(device),
                                (hidden & valid).to(device))
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                           float(config["gradient_clip"]))
            optimizer.step()
        totals += float(loss.detach()) * len(rows)
        seen += len(rows)
    return totals / max(seen, 1)


@torch.no_grad()
def branch_statistics(model, patch, split, cache, config, device, seed):
    """RMS/variance per branch and a collapse check, on one deterministic pass."""
    generator = torch.Generator().manual_seed(seed)
    anchors = patch.split_anchors[split]
    index = torch.arange(len(anchors))
    rows = anchors[index.numpy()]
    raw = torch.from_numpy(patch.X[rows])
    valid = torch.from_numpy(patch.M[rows])
    masked, hidden, visible = mask_views(raw, valid, config["hidden_fraction"],
                                         generator=generator)
    model.eval()
    physics = model.physics_latent(masked.to(device), visible.to(device))
    outputs = model(physics, cache[index].to(device), hidden.to(device))
    p = outputs["predicted_physics"].cpu().numpy()
    i = outputs["predicted_instrument"].cpu().numpy()
    m = valid.numpy()

    def rms(a):
        return float(np.sqrt((a[m] ** 2).mean()))

    # A collapsed head is nearly the same curve for every star, or nearly zero.
    across_stars = lambda a: float(np.mean(np.std(a, axis=0)))
    within_curve = lambda a: float(np.mean(np.std(a, axis=1)))
    return {
        "physics_rms": rms(p), "instrument_rms": rms(i),
        "physics_variance": float(np.var(p[m])), "instrument_variance": float(np.var(i[m])),
        "physics_std_across_stars": across_stars(p),
        "instrument_std_across_stars": across_stars(i),
        "physics_std_within_curve": within_curve(p),
        "instrument_std_within_curve": within_curve(i),
        "instrument_to_physics_rms_ratio": rms(i) / max(rms(p), 1e-9),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True,
                        help="Stage-1 best.pt; required so the wrong model cannot load")
    parser.add_argument("--config", default=os.path.join(HERE, "config_additive_heads.yaml"))
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = pick_device(config.get("device", "auto"))
    source_dir = os.path.dirname(os.path.abspath(args.source_checkpoint))
    out_dir = args.output_dir or os.path.join(os.path.dirname(source_dir), "additive_heads")
    os.makedirs(out_dir, exist_ok=True)

    model = AdditiveHeadsModel(args.source_checkpoint, map_location="cpu").to(device)
    before = dict(model.frozen_hashes)
    print(f"source checkpoint: {os.path.abspath(args.source_checkpoint)}")
    print(f"frozen encoder hashes before: {before}")
    print(f"parameters: {model.parameter_count()}", flush=True)
    assert not any(p.requires_grad for p in model.physics_encoder.parameters())
    assert not any(p.requires_grad for p in model.instrument_encoder.parameters())

    patch = build_patch(model.source_config, args.parquet, args.require_cross_sector)
    datasets = {name: CrossSectorAnchorDataset(patch, name, seed=config["seed"])
                for name in ("train", "val", "test")}
    from torch.utils.data import DataLoader
    audit_batch(patch, next(iter(DataLoader(datasets["train"],
                                            batch_size=config["anchors_per_step"],
                                            shuffle=True, drop_last=True))))

    caches = {name: cache_instrument(model, patch, name, device)
              for name in ("train", "val", "test")}
    print(f"cached instrument representations: "
          f"{ {k: tuple(v.shape) for k, v in caches.items()} }", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    history_path = os.path.join(out_dir, "history.csv")
    handle = open(history_path, "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])

    best = {"val": float("inf"), "epoch": -1}
    since_best, started, stop_reason = 0, time.time(), "max_epochs"
    for epoch in range(int(config["max_epochs"])):
        epoch_start = time.time()
        generator = torch.Generator().manual_seed(config["seed"] * 1000 + epoch)
        train_loss = run_split(model, patch, "train", caches["train"], config, device,
                               int(config["max_train_steps_per_epoch"]), generator,
                               optimizer)
        val_generator = torch.Generator().manual_seed(config["seed"])
        val_loss = run_split(model, patch, "val", caches["val"], config, device,
                             int(config["max_val_steps_per_epoch"]), val_generator)
        writer.writerow([epoch, train_loss, val_loss, round(time.time() - epoch_start, 2)])
        handle.flush()
        print(f"epoch {epoch:2d}: train {train_loss:.4f} | val {val_loss:.4f} "
              f"| {time.time() - epoch_start:.1f}s", flush=True)
        if val_loss < best["val"]:
            best = {"val": val_loss, "epoch": epoch, "train": train_loss}
            since_best = 0
            torch.save({"physics_head": model.physics_head.state_dict(),
                        "instrument_head": model.instrument_head.state_dict(),
                        "source_checkpoint": os.path.abspath(args.source_checkpoint),
                        "source_config": model.source_config, "config": config,
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
    test_loss = run_split(model, patch, "test", caches["test"], config, device,
                          int(config["max_val_steps_per_epoch"]), test_generator)
    controls = {}
    for name in ("physics", "instrument"):
        generator = torch.Generator().manual_seed(config["seed"])
        controls[f"shuffled_{name}_latents"] = run_split(
            model, patch, "test", caches["test"], config, device,
            int(config["max_val_steps_per_epoch"]), generator, shuffle=name)
    stats = branch_statistics(model, patch, "test", caches["test"], config, device,
                              config["seed"])

    after = {"physics_s4d": state_hash(model.physics_encoder),
             "instrument_s4d": state_hash(model.instrument_encoder)}
    unchanged = all(before[k] == after[k] for k in before)
    assert unchanged, f"frozen encoder weights changed: {before} -> {after}"

    collapsed = {
        "physics_head": bool(stats["physics_std_across_stars"] < 1e-3
                             or stats["physics_rms"] < 1e-3),
        "instrument_head": bool(stats["instrument_std_across_stars"] < 1e-3
                                or stats["instrument_rms"] < 1e-3),
    }
    summary = {
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "source_config": model.source_config, "config": config,
        "anchors": {k: int(len(v)) for k, v in patch.split_anchors.items()},
        "parameters": model.parameter_count(),
        "best_epoch": best["epoch"], "best_train_loss": best.get("train"),
        "best_val_loss": best["val"], "test_loss": test_loss,
        "stop_reason": stop_reason,
        "shuffle_controls": controls,
        "branch_statistics": stats,
        "collapsed": collapsed,
        "encoder_hashes_before": before, "encoder_hashes_after": after,
        "encoders_unchanged": bool(unchanged),
        "runtime_minutes": round((time.time() - started) / 60, 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\nbest epoch {best['epoch']}: train {best.get('train'):.4f} "
          f"val {best['val']:.4f} | test {test_loss:.4f}")
    print(f"shuffle controls: {controls}")
    print(f"branch stats: {stats}")
    print(f"collapsed: {collapsed}")
    print(f"encoders unchanged: {unchanged} ({after})")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
