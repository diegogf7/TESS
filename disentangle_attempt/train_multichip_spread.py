"""Train the p64/i4 two-decoder model over many chips (the GPU stage).

This is the completed Sector-1 / camera-4 / CCD-2 experiment run on 80 chips.  The
model, the masking, the loss and the output equations are imported from the modules the
single-chip run used, so the scale-up cannot silently change them::

    SmoothL1(physics_decoder(masked_anchor) + instrument_decoder(eight_peers),
             raw_anchor)[hidden & valid]

The architecture check, gradient audit, tiny-overfit gate, collapse diagnostics and
held-out plotting are reused verbatim from ``train_twodecoder_spread``.  Those helpers
are chip-agnostic, and sharing them is deliberate: the scaled run is verified by the
same code that verified the single-chip run, including its exact 1,777,764-parameter
assertion, which still holds because the architecture is unchanged.

What this module adds is only what more data requires: a cached multi-chip patch, an
epoch defined as a fixed number of steps rather than a pass over anchors, a capped
per-epoch validation subset, and a per-chip breakdown of the final test metrics.

    python -m disentangle_attempt.train_multichip_spread \\
        --config disentangle_attempt/config_multichip_s1_s5.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from disentangle_attempt.multichip_spread_peers import SPLITS, CachedMultiChipPatch
from disentangle_attempt.train import pick_device, set_seed
from disentangle_attempt.twodecoder_spread_model import build_twodecoder_model

# Reused verbatim from the single-chip trainer so the scaled run is checked and scored
# by identical code.  These are private only in the sense of "not a public API".
from disentangle_attempt.train_twodecoder_spread import (
    _architecture_checks,
    _collapse_flags,
    _complementary_inference,
    _evaluate_loader,
    _forward_batch,
    _gradient_norms,
    _jsonable,
    _per_star_metrics,
    _plot_heldout,
    _plot_training_history,
    _synchronize,
    _tiny_overfit_gate,
    HISTORY_FIELDS,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config_multichip_s1_s5.yaml"

TRAINING_ARTIFACTS = (
    "config.yaml",
    "architecture_checks.json",
    "model_summary.txt",
    "tiny_overfit.json",
    "training_history.csv",
    "training_curves.png",
    "best.pt",
    "last.pt",
    "metrics.json",
    "per_chip_test_metrics.csv",
    "heldout_per_star.csv",
    "plots",
)


class MultiChipAnchorDataset(Dataset):
    """One item = one anchor bundle, read from the memory-mapped cache.

    Deliberately mirrors ``CrossSectorAnchorDataset``; the only difference is an
    explicit copy out of the memory map, so a batch never holds a view into a ~1 GB
    mapping across worker processes.
    """

    def __init__(self, patch: CachedMultiChipPatch, split: str, seed: int = 0):
        self.patch = patch
        self.split = split
        self.anchors = patch.split_anchors[split]
        self.peer_rows, self.peer_distance = patch.peers[split]
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patch = self.patch
        anchor = int(self.anchors[index])
        peers = np.asarray(self.peer_rows[index], dtype=np.int64)
        return {
            "anchor_raw": torch.from_numpy(np.array(patch.X[anchor], dtype=np.float32)),
            "anchor_valid_mask": torch.from_numpy(np.array(patch.M[anchor], dtype=bool)),
            "peer_raw": torch.from_numpy(np.array(patch.X[peers], dtype=np.float32)),
            "peer_mask": torch.from_numpy(np.array(patch.M[peers], dtype=bool)),
            "anchor_tic_ids": torch.tensor(int(patch.tic_int[anchor]), dtype=torch.int64),
            "anchor_sector": torch.tensor(int(patch.sector[anchor]), dtype=torch.int64),
            "peer_tic_ids": torch.from_numpy(np.array(patch.tic_int[peers], dtype=np.int64)),
            "peer_distances": torch.from_numpy(
                np.array(self.peer_distance[index], dtype=np.float32)
            ),
            "anchor_row": torch.tensor(anchor, dtype=torch.int64),
            "peer_rows": torch.from_numpy(peers),
        }


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _validate_config(config: dict[str, Any]) -> None:
    """Refuse changes to the parts of the experiment the scale-up must not touch."""
    fixed = {
        "curve_length": 1024,
        "n_peers": 8,
        "n_tokens": 32,
        "token_dim": 16,
        "physics_latent_dim": 64,
        "instrument_token_dim": 4,
        "instrument_context_dim": 32,
        "d_model": 128,
        "n_layers": 4,
        "hidden_fraction": 0.25,
        "mask_style": "contiguous_windows",
        "peer_selection": "spread_two_band",
        "peers_per_distance_band": 4,
        "optimizer": "adamw",
        "scheduler": "constant",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
    }
    wrong = {
        key: (config.get(key), want)
        for key, want in fixed.items()
        if config.get(key) != want
    }
    if wrong:
        raise ValueError(f"configuration changes the fixed experiment: {wrong}")
    bands = np.asarray(config.get("peer_distance_bands_px"), dtype=float)
    if bands.shape != (2, 2) or not np.array_equal(
        bands, np.asarray([[128.0, 384.0], [384.0, 768.0]])
    ):
        raise ValueError("peer bands must remain [128,384) and [384,768]")
    tiers = [float(v) for v in config.get("minimum_peer_separation_tiers_px", [])]
    if tiers != [256.0, 192.0, 128.0]:
        raise ValueError("peer-spacing fallbacks must be exactly 256, 192, then 128 px")


def _refuse_collisions(output_dir: Path) -> None:
    present = [str(output_dir / n) for n in TRAINING_ARTIFACTS if (output_dir / n).exists()]
    if present:
        raise FileExistsError(
            "refusing to overwrite artifacts from an existing training attempt: "
            + ", ".join(present)
        )


def _capped_loader(
    dataset: Dataset, cap: int, batch_size: int, workers: int, seed: int
) -> DataLoader:
    """A fixed deterministic subset, so per-epoch validation is cheap and comparable."""
    if cap <= 0 or cap >= len(dataset):
        subset: Dataset = dataset
    else:
        picked = np.random.default_rng(seed).permutation(len(dataset))[:cap]
        subset = Subset(dataset, np.sort(picked).tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=workers)


def _per_chip_metrics(
    model: torch.nn.Module,
    patch: CachedMultiChipPatch,
    dataset: MultiChipAnchorDataset,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> pd.DataFrame:
    """Final test loss broken out per chip -- does the model work everywhere?"""
    chips = np.asarray(
        [patch.chip_name_of_row(int(row)) for row in dataset.anchors]
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(set(chips.tolist())):
        indices = np.flatnonzero(chips == name).tolist()
        if not indices:
            continue
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        metrics = _evaluate_loader(model, loader, config, device, seed=seed)
        rows.append(
            {
                "chip": name,
                "anchors": len(indices),
                "masked_smooth_l1": metrics["masked_smooth_l1"],
                "physics_output_rms": metrics["physics_output_rms"],
                "instrument_output_rms": metrics["instrument_output_rms"],
                "residual_rms": metrics["residual_rms"],
                "branch_cosine": metrics["branch_cosine"],
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.config.expanduser().resolve().open() as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve(config["output_dir"])
    )
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / "dataset_cache"
    )
    summary_path = output_dir / "geometry_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"no geometry summary at {summary_path}; run "
            "disentangle_attempt.prepare_multichip_dataset first"
        )
    geometry_summary = json.loads(summary_path.read_text())
    if geometry_summary.get("status") != "PASS":
        raise RuntimeError(
            f"training is blocked: geometry status is {geometry_summary.get('status')!r}"
        )
    if config.get("refuse_existing_training_artifacts", True):
        _refuse_collisions(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch = CachedMultiChipPatch(cache_dir)
    datasets = {
        split: MultiChipAnchorDataset(patch, split, seed=int(config["seed"]))
        for split in SPLITS
    }
    counts = {split: len(datasets[split]) for split in SPLITS}
    if min(counts.values()) == 0:
        raise RuntimeError(f"a split has no anchors: {counts}")

    set_seed(int(config["seed"]))
    device = pick_device(args.device or config.get("device", "auto"))
    use_amp = bool(config.get("mixed_precision")) and device.type == "cuda"
    workers = int(config.get("num_workers", 0))
    if workers and device.type == "mps":
        workers = 0
    batch_size = int(config["anchors_per_step"])
    print(
        f"device {device} | mixed precision {'on' if use_amp else 'off'} | "
        f"{len(patch.chips)} chips | anchors {counts}",
        flush=True,
    )

    saved_config = dict(config)
    saved_config.update(
        {"output_dir": str(output_dir), "cache_dir": str(cache_dir)}
    )
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(saved_config, handle, sort_keys=False)

    # --- architecture and gradient audit, using the single-chip run's own checker ---
    audit_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=False, num_workers=workers
    )
    audit_batch = next(iter(audit_loader))
    set_seed(int(config["seed"]))
    audit_model = build_twodecoder_model(config).to(device)
    architecture = _architecture_checks(audit_model, audit_batch, config, device)
    architecture["geometry_summary"] = geometry_summary
    architecture["anchor_counts"] = counts
    architecture["chips"] = len(patch.chips)
    with (output_dir / "architecture_checks.json").open("w") as handle:
        json.dump(_jsonable(architecture), handle, indent=2)
        handle.write("\n")
    del audit_model

    tiny = _tiny_overfit_gate(datasets["train"], config, device, use_amp)
    with (output_dir / "tiny_overfit.json").open("w") as handle:
        json.dump(_jsonable(tiny), handle, indent=2)
        handle.write("\n")
    print(
        f"tiny-overfit PASS at step {tiny['passed_at_step']}: "
        f"loss ratio {tiny['best_to_initial_ratio']:.4f}",
        flush=True,
    )

    set_seed(int(config["seed"]))
    model = build_twodecoder_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler_generator = torch.Generator().manual_seed(int(config["seed"]) + 50000)
    steps_per_epoch = int(config["max_train_steps_per_epoch"])
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        drop_last=True,
        num_workers=workers,
        sampler=RandomSampler(
            datasets["train"],
            replacement=True,
            num_samples=batch_size * steps_per_epoch,
            generator=sampler_generator,
        ),
    )
    epoch_val_loader = _capped_loader(
        datasets["val"],
        int(config.get("max_val_anchors_per_epoch", 0)),
        batch_size,
        workers,
        int(config["seed"]) + 90000,
    )

    history_path = output_dir / "training_history.csv"
    history_handle = history_path.open("w", newline="")
    writer = csv.DictWriter(history_handle, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    history_handle.flush()

    started = time.time()
    best = {"loss": float("inf"), "epoch": -1}
    since_best = 0
    collapse_streak = {"physics": 0, "instrument": 0}
    stop_reason = "max_epochs"
    final_epoch = -1

    for epoch in range(1, int(config["max_epochs"]) + 1):
        datasets["train"].set_epoch(epoch)
        epoch_started = time.time()
        model.train()
        mask_generator = torch.Generator().manual_seed(int(config["seed"]) * 1000 + epoch)
        loss_numerator, loss_denominator = 0.0, 0
        gradient_sums = {
            name: 0.0
            for name in (
                "physics_branch", "instrument_branch", "physics_s4d",
                "physics_projection", "physics_decoder", "instrument_s4d",
                "instrument_projection", "instrument_decoder",
            )
        }
        physics_square = torch.zeros((), device=device)
        instrument_square = torch.zeros((), device=device)
        valid_count = torch.zeros((), device=device)
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, outputs, loss_mask = _forward_batch(
                    model, batch, config, device, generator=mask_generator
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_values = _gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            count = int(loss_mask.sum().detach().cpu())
            loss_numerator += float(loss.detach().cpu()) * count
            loss_denominator += count
            for name in gradient_sums:
                gradient_sums[name] += gradient_values[name]
            output_valid = batch["anchor_valid_mask"].to(device=device, dtype=torch.float32)
            physics_square += (
                outputs["physics_curve"].detach().float().square() * output_valid
            ).sum()
            instrument_square += (
                outputs["instrument_correction"].detach().float().square() * output_valid
            ).sum()
            valid_count += output_valid.sum()
            steps += 1

        train_loss = loss_numerator / max(loss_denominator, 1)
        train_physics_rms = float(torch.sqrt(physics_square / valid_count.clamp(min=1)).cpu())
        train_instrument_rms = float(
            torch.sqrt(instrument_square / valid_count.clamp(min=1)).cpu()
        )
        val = _evaluate_loader(
            model, epoch_val_loader, config, device, seed=int(config["seed"]) + 60000
        )
        collapse_now = _collapse_flags(val, config)
        for branch in collapse_streak:
            collapse_streak[branch] = (
                collapse_streak[branch] + 1 if collapse_now[branch] else 0
            )
        _synchronize(device)
        epoch_seconds = time.time() - epoch_started
        wall_minutes = (time.time() - started) / 60.0
        gradient_means = {n: v / max(steps, 1) for n, v in gradient_sums.items()}
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["masked_smooth_l1"],
                "train_physics_output_rms": train_physics_rms,
                "train_instrument_output_rms": train_instrument_rms,
                "val_physics_output_rms": val["physics_output_rms"],
                "val_instrument_output_rms": val["instrument_output_rms"],
                "val_reconstructed_output_rms": val["reconstructed_output_rms"],
                "val_residual_rms": val["residual_rms"],
                "val_physics_across_anchor_variation": val["physics_across_anchor_variation"],
                "val_instrument_across_anchor_variation": val["instrument_across_anchor_variation"],
                "val_physics_temporal_variation": val["physics_temporal_variation"],
                "val_instrument_temporal_variation": val["instrument_temporal_variation"],
                "val_branch_cosine": val["branch_cosine"],
                "val_cancellation_ratio": val["cancellation_ratio"],
                "physics_branch_gradient_norm": gradient_means["physics_branch"],
                "instrument_branch_gradient_norm": gradient_means["instrument_branch"],
                "physics_s4d_gradient_norm": gradient_means["physics_s4d"],
                "physics_projection_gradient_norm": gradient_means["physics_projection"],
                "physics_decoder_gradient_norm": gradient_means["physics_decoder"],
                "instrument_s4d_gradient_norm": gradient_means["instrument_s4d"],
                "instrument_projection_gradient_norm": gradient_means["instrument_projection"],
                "instrument_decoder_gradient_norm": gradient_means["instrument_decoder"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "wall_minutes": wall_minutes,
                "physics_collapse_streak": collapse_streak["physics"],
                "instrument_collapse_streak": collapse_streak["instrument"],
            }
        )
        history_handle.flush()
        final_epoch = epoch
        print(
            f"epoch {epoch:02d} | train {train_loss:.6f} | val {val['masked_smooth_l1']:.6f} | "
            f"physics RMS {val['physics_output_rms']:.4f} | "
            f"instrument RMS {val['instrument_output_rms']:.4f} | "
            f"{epoch_seconds:.0f}s",
            flush=True,
        )

        checkpoint = {
            "format": "multichip_twodecoder_spread_v1",
            "model_class": "TwoDecoderSpreadModel",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": saved_config,
            "epoch": int(epoch),
            "validation_masked_smooth_l1": float(val["masked_smooth_l1"]),
            "chips": [list(chip) for chip in patch.chips],
            "objective": "hidden-valid masked Smooth-L1 on reconstructed_raw only",
            "output_equations": {
                "reconstructed_raw": "physics_curve + instrument_correction",
                "cleaned_curve": "raw_anchor - instrument_correction",
                "residual": "raw_anchor - reconstructed_raw",
            },
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val["masked_smooth_l1"] < best["loss"]:
            best = {"loss": val["masked_smooth_l1"], "epoch": epoch}
            since_best = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            since_best += 1

        collapsed = [
            name
            for name, streak in collapse_streak.items()
            if streak >= int(config["collapse_patience_epochs"])
        ]
        if collapsed:
            stop_reason = "branch_collapse:" + ",".join(collapsed)
            print(f"stopping for sustained branch collapse: {collapsed}", flush=True)
            break
        if since_best >= int(config["early_stopping_patience"]):
            stop_reason = "early_stopping"
            print(f"early stopping after epoch {epoch}", flush=True)
            break
        if wall_minutes >= float(config["max_runtime_minutes"]):
            stop_reason = "max_runtime"
            print("runtime limit reached", flush=True)
            break

    history_handle.close()
    training_seconds = time.time() - started
    _plot_training_history(history_path, output_dir / "training_curves.png")

    if best["epoch"] < 0 or not (output_dir / "best.pt").is_file():
        raise RuntimeError("training ended without a validation-selected checkpoint")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    # Full splits are scored once, here, after validation alone selected the epoch.
    full_loaders = {
        split: DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False, num_workers=workers
        )
        for split in SPLITS
    }
    final = {
        split: _evaluate_loader(
            model, full_loaders[split], config, device, seed=int(config["seed"]) + 70001 + index
        )
        for index, split in enumerate(SPLITS)
    }
    per_chip = _per_chip_metrics(
        model, patch, datasets["test"], config, device, batch_size, workers,
        seed=int(config["seed"]) + 80000,
    )
    per_chip.to_csv(output_dir / "per_chip_test_metrics.csv", index=False)

    heldout = _complementary_inference(model, full_loaders["test"], patch, config, device)
    np.savez_compressed(output_dir / "heldout_full_curve_arrays.npz", **heldout)
    _per_star_metrics(heldout).to_csv(output_dir / "heldout_per_star.csv", index=False)
    plot_manifest = _plot_heldout(
        heldout, output_dir / "plots", int(config["heldout_plot_count"])
    )
    plot_manifest.to_csv(output_dir / "plots" / "manifest.csv", index=False)

    final_collapse = _collapse_flags(final["val"], config)
    any_collapse = any(final_collapse.values()) or stop_reason.startswith("branch_collapse")
    history = pd.read_csv(history_path)
    metrics = {
        "run_name": config["run_name"],
        "status": "COMPLETE_WITH_COLLAPSE" if any_collapse else "COMPLETE",
        "device": str(device),
        "chips": len(patch.chips),
        "anchor_counts": counts,
        "geometry": geometry_summary,
        "parameters": model.parameter_count(),
        "best_epoch": int(best["epoch"]),
        "best_validation_masked_smooth_l1": float(best["loss"]),
        "final_full_split_evaluation": {
            "train": final["train"],
            "validation": final["val"],
            "test": final["test"],
            "test_evaluation_count_after_checkpoint_selection": 1,
        },
        "per_chip_test": {
            "chips_scored": int(len(per_chip)),
            "masked_smooth_l1_median": float(per_chip["masked_smooth_l1"].median())
            if len(per_chip) else None,
            "masked_smooth_l1_worst": float(per_chip["masked_smooth_l1"].max())
            if len(per_chip) else None,
            "masked_smooth_l1_best": float(per_chip["masked_smooth_l1"].min())
            if len(per_chip) else None,
        },
        "training": {
            "last_epoch": final_epoch,
            "minutes": training_seconds / 60.0,
            "stop_reason": stop_reason,
            "steps_per_epoch": steps_per_epoch,
            "meaningful_nonzero_gradients_both_branches": bool(
                (history["physics_branch_gradient_norm"] > 0).all()
                and (history["instrument_branch_gradient_norm"] > 0).all()
            ),
            "tiny_overfit": tiny,
        },
        "collapse_diagnostics": {
            "final_validation_flags": final_collapse,
            "either_output_branch_collapsed": any_collapse,
        },
        "identifiability": {
            "physically_proven_components": False,
            "statement": "These are learned additive components under asymmetric "
            "information routing. Reconstruction alone does not physically identify "
            "the decomposition; injection and peer-control tests remain future work.",
        },
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)
        handle.write("\n")

    print(
        f"\ncomplete: best epoch {best['epoch']} | "
        f"train {final['train']['masked_smooth_l1']:.6f} | "
        f"val {final['val']['masked_smooth_l1']:.6f} | "
        f"test {final['test']['masked_smooth_l1']:.6f} | "
        f"{training_seconds / 60.0:.1f} min",
        flush=True,
    )
    if len(per_chip):
        print(
            f"per-chip test masked Smooth-L1: median "
            f"{per_chip['masked_smooth_l1'].median():.5f}, "
            f"best {per_chip['masked_smooth_l1'].min():.5f}, "
            f"worst {per_chip['masked_smooth_l1'].max():.5f} "
            f"over {len(per_chip)} chips",
            flush=True,
        )
    print(
        "The named branches are learned additive outputs, not yet physically proven "
        "components.",
        flush=True,
    )
    print(f"outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
