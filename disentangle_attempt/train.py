"""Fast training run for the cross-sector disentangle attempt.

Each optimizer step averages the loss over 32 anchors and calls backward() exactly
once. The reconstruction gradient reaches the physics S4D, the instrument S4D and the
decoder; the cross-sector consistency gradient reaches the shared physics S4D only.

After training, the best checkpoint is re-scored under three branch-use controls
(shuffled physics inputs, random peers, wrong cross-sector TIC) and the quiet
reference context is selected and saved.

    python -m disentangle_attempt.train --config disentangle_attempt/config_fast.yaml
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
from torch.utils.data import DataLoader, RandomSampler

from disentangle_attempt.dataset import (CrossSectorAnchorDataset, CrossSectorPatch,
                                        audit_batch)
from disentangle_attempt.losses import total_loss, visible_reconstruction
from disentangle_attempt.masking import mask_views
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import (build_reference_context,
                                                   save_reference_context)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_fast.yaml")
DEFAULT_PARQUET = os.path.join(HERE, "data", "cross_sector_raw.parquet")

BATCH_KEYS = ("anchor_raw", "anchor_valid_mask", "other_sector_raw", "other_sector_mask",
              "peer_raw", "peer_mask")


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested="auto"):
    if requested not in ("auto", None):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ------------------------------------------------------------------ one forward
def forward_batch(model, batch, config, generator=None, condition=None, patch=None,
                  split=None, rng=None, device=torch.device("cpu")):
    """Mask on CPU (device-independent RNG), optionally corrupt one branch, forward.

    The reconstruction target, its validity and the hidden mask always belong to the
    TRUE anchor, so every control keeps the identical loss geometry and only the
    contents of one branch change.
    """
    anchor_raw = batch["anchor_raw"]
    anchor_valid = batch["anchor_valid_mask"]
    masked, hidden, visible = mask_views(anchor_raw, anchor_valid,
                                         config["hidden_fraction"], generator=generator)

    physics_raw, physics_valid = anchor_raw, anchor_valid
    peer_raw, peer_mask = batch["peer_raw"], batch["peer_mask"]
    other_raw, other_mask = batch["other_sector_raw"], batch["other_sector_mask"]

    if condition == "shuffle_physics":
        roll = torch.roll(torch.arange(len(anchor_raw)), 1)      # another TIC's curve
        physics_raw, physics_valid = anchor_raw[roll], anchor_valid[roll]
    elif condition == "random_peers":
        rows = patch.random_peer_rows(batch["anchor_row"].numpy(), split, rng)
        peer_raw = torch.from_numpy(patch.X[rows])
        peer_mask = torch.from_numpy(patch.M[rows])
    elif condition == "wrong_other_sector":
        roll = torch.roll(torch.arange(len(other_raw)), 1)
        other_raw, other_mask = other_raw[roll], other_mask[roll]
    elif condition is not None:
        raise ValueError(f"unknown branch-use condition {condition!r}")

    if condition == "shuffle_physics":                            # same mask geometry
        masked_physics = physics_raw.masked_fill(hidden, 0.0)
        physics_visible = physics_valid & ~hidden
    else:
        masked_physics, physics_visible = masked, visible

    tensors = [masked_physics, physics_visible, peer_raw, peer_mask, hidden,
               other_raw, other_mask, anchor_raw, anchor_valid]
    (masked_physics, physics_visible, peer_raw, peer_mask, hidden, other_raw,
     other_mask, anchor_raw, anchor_valid) = [t.to(device) for t in tensors]

    outputs = model(masked_physics, physics_visible, peer_raw, peer_mask, hidden,
                    other_raw, other_mask)
    loss, parts = total_loss(outputs, anchor_raw, anchor_valid,
                             config["physics_consistency_weight"])
    parts["visible_reconstruction"] = visible_reconstruction(outputs, anchor_raw, anchor_valid)
    return loss, parts, outputs


def evaluate(model, loader, config, device, seed, condition=None, patch=None,
             split=None, max_steps=None):
    """Deterministic evaluation: identical batches under every condition."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    totals, n = {}, 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            _, parts, _ = forward_batch(model, batch, config, generator, condition,
                                        patch, split, rng, device)
            weight = len(batch["anchor_raw"])
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * weight
            n += weight
    model.train()
    return {key: value / max(n, 1) for key, value in totals.items()}


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config.get("run_name") or time.strftime("fast_%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    if config.get("mask_style", "contiguous_windows") != "contiguous_windows":
        raise ValueError("only mask_style=contiguous_windows is implemented")
    if config.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("only optimizer=adamw is implemented")
    set_seed(config["seed"])
    device = pick_device(config.get("device", "auto"))

    # Mixed precision would run the S4D FFT kernels in fp16; complex half support is
    # CUDA-only, so it is enabled there and reported as disabled elsewhere.
    use_amp = bool(config.get("mixed_precision")) and device.type == "cuda"
    print(f"device {device} | mixed precision {'on' if use_amp else 'off'}", flush=True)

    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=config.get("sector", "auto"), camera=config.get("camera", "auto"),
        ccd=config.get("ccd", "auto"), curve_length=config["curve_length"],
        n_peers=config["n_peers"], min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"))
    eligibility = patch.eligibility_table()
    eligibility.to_csv(os.path.join(out_dir, "eligibility.csv"), index=False)

    datasets = {name: CrossSectorAnchorDataset(patch, name, seed=config["seed"])
                for name in ("train", "val", "test")}
    batch = int(config["anchors_per_step"])
    workers = int(config.get("num_workers", 0))
    if workers and device.type == "mps":
        workers = 0                      # curves are already in RAM; workers only copy
    train_loader = DataLoader(
        datasets["train"], batch_size=batch, drop_last=True, num_workers=workers,
        sampler=RandomSampler(datasets["train"], replacement=True,
                              num_samples=batch * int(config["max_train_steps_per_epoch"])))
    val_loader = DataLoader(datasets["val"], batch_size=batch, shuffle=False,
                            num_workers=workers)

    audit_batch(patch, next(iter(DataLoader(datasets["train"], batch_size=batch,
                                            shuffle=True, drop_last=True))))

    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4),
                             dropout=config.get("dropout", 0.0),
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    counts = model.parameter_count()
    print(f"parameters: {counts}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_path = os.path.join(out_dir, "history.csv")
    history_file = open(history_path, "w", newline="")
    writer = csv.writer(history_file)
    writer.writerow(["epoch", "train_total", "train_reconstruction", "train_sector_consistency",
                     "val_total", "val_reconstruction", "val_sector_consistency",
                     "val_visible_reconstruction", "seconds", "wall_minutes"])

    start = time.time()
    best = {"val_reconstruction": float("inf"), "epoch": -1}
    patience = int(config["early_stopping_patience"])
    since_best = 0
    stop_reason = "max_epochs"

    for epoch in range(int(config["max_epochs"])):
        datasets["train"].set_epoch(epoch)
        epoch_start = time.time()
        sums, steps = {}, 0
        generator = torch.Generator().manual_seed(config["seed"] * 1000 + epoch)
        for step_batch in train_loader:
            with torch.autocast("cuda", enabled=use_amp):
                loss, parts, _ = forward_batch(model, step_batch, config, generator,
                                               device=device)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            steps += 1
        train_metrics = {k: v / max(steps, 1) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, config, device, seed=config["seed"],
                               max_steps=config.get("max_val_steps_per_epoch"))

        elapsed = time.time() - start
        writer.writerow([epoch, train_metrics["total"], train_metrics["reconstruction"],
                         train_metrics["sector_consistency"], val_metrics["total"],
                         val_metrics["reconstruction"], val_metrics["sector_consistency"],
                         val_metrics["visible_reconstruction"],
                         round(time.time() - epoch_start, 2), round(elapsed / 60, 2)])
        history_file.flush()
        print(f"epoch {epoch}: train recon {train_metrics['reconstruction']:.4f} "
              f"cons {train_metrics['sector_consistency']:.4f} | val recon "
              f"{val_metrics['reconstruction']:.4f} cons {val_metrics['sector_consistency']:.4f} "
              f"| {time.time() - epoch_start:.1f}s", flush=True)

        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                   os.path.join(out_dir, "last.pt"))
        if val_metrics["reconstruction"] < best["val_reconstruction"]:
            best = {"val_reconstruction": val_metrics["reconstruction"], "epoch": epoch,
                    "val_sector_consistency": val_metrics["sector_consistency"]}
            since_best = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "target": patch.target}, os.path.join(out_dir, "best.pt"))
        else:
            since_best += 1
            if since_best >= patience:
                stop_reason = "early_stopping"
                print(f"early stopping after epoch {epoch}", flush=True)
                break
        if elapsed / 60 >= float(config["max_runtime_minutes"]):
            stop_reason = "max_runtime"
            print("wall-clock limit reached", flush=True)
            break
    history_file.close()
    train_seconds = time.time() - start

    # ------------------------------------------------------------ branch-use tests
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     weights_only=False)["model"])
    baseline = evaluate(model, val_loader, config, device, seed=config["seed"],
                        max_steps=config.get("max_val_steps_per_epoch"))
    branch = {"baseline": baseline, "conditions": {}}
    for condition in ("shuffle_physics", "random_peers", "wrong_other_sector"):
        scores = evaluate(model, val_loader, config, device, seed=config["seed"],
                          condition=condition, patch=patch, split="val",
                          max_steps=config.get("max_val_steps_per_epoch"))
        branch["conditions"][condition] = {
            "metrics": scores,
            "delta_reconstruction": scores["reconstruction"] - baseline["reconstruction"],
            "delta_sector_consistency": (scores["sector_consistency"]
                                         - baseline["sector_consistency"]),
        }
    branch["verdicts"] = {
        "physics_branch_used": bool(branch["conditions"]["shuffle_physics"]["delta_reconstruction"] > 0),
        "instrument_branch_used": bool(branch["conditions"]["random_peers"]["delta_reconstruction"] > 0),
        "cross_sector_branch_used": bool(
            branch["conditions"]["wrong_other_sector"]["delta_sector_consistency"] > 0),
    }
    with open(os.path.join(out_dir, "branch_use_tests.json"), "w") as handle:
        json.dump(branch, handle, indent=2)
    for name, result in branch["conditions"].items():
        print(f"branch-use {name}: d_recon {result['delta_reconstruction']:+.4f} "
              f"d_consistency {result['delta_sector_consistency']:+.4f}", flush=True)

    # -------------------------------------------------------- quiet reference set
    reference = build_reference_context(patch, split="train", n_peers=config["n_peers"])
    reference_meta = save_reference_context(patch, reference, out_dir)

    test_metrics = evaluate(model, DataLoader(datasets["test"], batch_size=batch,
                                              shuffle=False, num_workers=workers),
                            config, device, seed=config["seed"])
    metrics = {
        "run_name": run_name, "device": str(device), "mixed_precision": use_amp,
        "parameters": counts,
        "target_sector_camera_ccd": {"sector": patch.target[0], "camera": patch.target[1],
                                     "ccd": patch.target[2]},
        "eligible_cross_sector_tics": int(len(patch.eligible_rows)),
        "anchors": {k: int(len(v)) for k, v in patch.split_anchors.items()},
        "peer_pool": {k: int(len(v)) for k, v in patch.split_pool.items()},
        "training_seconds": round(train_seconds, 1),
        "training_minutes": round(train_seconds / 60, 2),
        "stop_reason": stop_reason,
        "best_epoch": best["epoch"],
        "best_val_reconstruction": best["val_reconstruction"],
        "best_val_sector_consistency": best.get("val_sector_consistency"),
        "test": test_metrics,
        "branch_use_tests": branch,
        "reference_context": reference_meta,
        "config": config,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(out_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"\nrun {run_name}: best epoch {best['epoch']}, val reconstruction "
          f"{best['val_reconstruction']:.4f}, {train_seconds / 60:.1f} min, "
          f"stop reason {stop_reason}", flush=True)
    print(f"outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
