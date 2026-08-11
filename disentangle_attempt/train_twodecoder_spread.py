"""Train and evaluate the isolated p64/i4 two-decoder spread-peer model.

The output directory is intentionally a staged run directory.  Acquisition and the
literal geometry audit happen first and may leave ``peer_data/``,
``peer_selection.csv``, ``peer_geometry_summary.json`` and ``peer_geometry/`` there.
This trainer accepts those artifacts, verifies them, and refuses to overwrite any
artifact produced by a previous training attempt.

The sole optimization target is the unmodified, normalized raw anchor on cadences
that are both valid and hidden from the physics encoder::

    SmoothL1(physics_decoder(masked_anchor) + instrument_decoder(eight_peers),
             raw_anchor)[hidden & valid]

No target or correction is constructed from peer statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, default_collate

from disentangle_attempt.dataset import CrossSectorAnchorDataset
from disentangle_attempt.losses import masked_smooth_l1
from disentangle_attempt.masking import complementary_masks, mask_views
from disentangle_attempt.train import pick_device, set_seed
from disentangle_attempt.twodecoder_spread_model import build_twodecoder_model


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config_local_twodecoder_spread.yaml"
EXPECTED_RUN_NAME = "local_s1_c4_ccd2_twodecoder_p64_i4_spread"
EXPECTED_SPLIT_COUNTS = {"train": 320, "val": 40, "test": 40}
SPLITS = ("train", "val", "test")

TRAINING_ARTIFACTS = (
    "config.yaml",
    "split_tics.json",
    "peer_selection_attempts.csv",
    "peer_pool_assignment.csv",
    "supplemental_ingestion_audit.csv",
    "architecture_checks.json",
    "model_summary.txt",
    "tiny_overfit.json",
    "training_history.csv",
    "training_curves.png",
    "best.pt",
    "last.pt",
    "metrics.json",
    "heldout_full_curve_arrays.npz",
    "heldout_per_star.csv",
    "plots",
)

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_physics_output_rms",
    "train_instrument_output_rms",
    "val_physics_output_rms",
    "val_instrument_output_rms",
    "val_reconstructed_output_rms",
    "val_residual_rms",
    "val_physics_across_anchor_variation",
    "val_instrument_across_anchor_variation",
    "val_physics_temporal_variation",
    "val_instrument_temporal_variation",
    "val_branch_cosine",
    "val_cancellation_ratio",
    "physics_branch_gradient_norm",
    "instrument_branch_gradient_norm",
    "physics_s4d_gradient_norm",
    "physics_projection_gradient_norm",
    "physics_decoder_gradient_norm",
    "instrument_s4d_gradient_norm",
    "instrument_projection_gradient_norm",
    "instrument_decoder_gradient_norm",
    "learning_rate",
    "epoch_seconds",
    "wall_minutes",
    "physics_collapse_streak",
    "instrument_collapse_streak",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return result


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "run_name": EXPECTED_RUN_NAME,
        "seed": 42,
        "curve_length": 1024,
        "anchors_per_step": 32,
        "n_peers": 8,
        "sector": 1,
        "camera": 4,
        "ccd": 2,
        "max_eligible_anchors": 400,
        "physics_latent_dim": 64,
        "instrument_token_dim": 4,
        "instrument_context_dim": 32,
        "max_train_steps_per_epoch": 40,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "max_runtime_minutes": 45,
    }
    disagreements = {
        key: (config.get(key), wanted)
        for key, wanted in expected.items()
        if config.get(key) != wanted
    }
    if disagreements:
        raise ValueError(f"configuration changes the fixed experiment: {disagreements}")
    if config.get("expected_anchor_split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("expected anchor split must be exactly 320/40/40")
    if config.get("mask_style") != "contiguous_windows":
        raise ValueError("the experiment requires contiguous-window masking")
    if float(config.get("hidden_fraction", -1)) != 0.25:
        raise ValueError("the experiment requires a 25% hidden fraction")
    if config.get("optimizer", "").lower() != "adamw":
        raise ValueError("the experiment requires AdamW")
    if float(config.get("learning_rate", -1)) != 3e-4:
        raise ValueError("learning rate must remain 3e-4")
    if float(config.get("weight_decay", -1)) != 1e-4:
        raise ValueError("weight decay must remain 1e-4")
    if config.get("scheduler") != "constant":
        raise ValueError("the baseline uses a constant learning rate")
    if float(config.get("gradient_clip", -1)) != 1.0:
        raise ValueError("gradient clipping must remain 1.0")
    if config.get("save_reference_context", True):
        raise ValueError("quiet/reference context is forbidden")
    if config.get("peer_selection") != "spread_two_band":
        raise ValueError("the trainer requires the literal two-band spread selector")
    bands = np.asarray(config.get("peer_distance_bands_px"), dtype=float)
    if bands.shape != (2, 2) or not np.array_equal(
        bands, np.asarray([[128.0, 384.0], [384.0, 768.0]])
    ):
        raise ValueError("peer bands must remain [128,384) and [384,768]")
    if config.get("peer_distance_band_upper_inclusive") != [False, True]:
        raise ValueError(
            "inner upper bound must be open and outer upper bound inclusive"
        )
    if int(config.get("peers_per_distance_band", -1)) != 4:
        raise ValueError("each distance band must supply exactly four peers")
    tiers = [
        float(value) for value in config.get("minimum_peer_separation_tiers_px", [])
    ]
    if tiers != [256.0, 192.0, 128.0]:
        raise ValueError("peer-spacing fallbacks must be exactly 256, 192, then 128 px")


def _verify_staged_geometry(output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = output_dir / "peer_geometry_summary.json"
    selection_path = output_dir / "peer_selection.csv"
    if not summary_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            "training is blocked until the new feasibility audit has saved "
            "peer_geometry_summary.json and peer_selection.csv"
        )
    with summary_path.open() as handle:
        summary = json.load(handle)
    if summary.get("status") != "PASS":
        raise RuntimeError(
            f"training is blocked because the geometry audit status is "
            f"{summary.get('status')!r}"
        )
    if summary.get("most_anchors_infeasible"):
        raise RuntimeError(
            "training is blocked: the audit marked most anchors infeasible"
        )
    if summary.get("contract_violations"):
        raise RuntimeError(
            f"training is blocked by geometry violations: "
            f"{summary['contract_violations']}"
        )
    selected = {
        split: int(summary.get("selected_anchor_counts", {}).get(split, -1))
        for split in SPLITS
    }
    if selected != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "the geometry audit did not retain the unchanged 320/40/40 anchors: "
            f"{selected}"
        )
    if not summary.get("determinism", {}).get("independent_rebuild_verified", False):
        raise RuntimeError("the geometry audit did not verify an independent rebuild")
    if not summary.get("geometry_plots", {}).get("all_true_equal_scale", False):
        raise RuntimeError(
            "the geometry audit plots are not verified at true equal scale"
        )
    selection = pd.read_csv(selection_path, dtype={"anchor_TIC": str, "peer_TIC": str})
    required = {
        "split",
        "anchor_TIC",
        "peer_order",
        "peer_TIC",
        "anchor_distance",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(f"audited peer selection is missing columns: {missing}")
    return summary, selection


def _refuse_training_collisions(output_dir: Path) -> None:
    collisions = [output_dir / name for name in TRAINING_ARTIFACTS]
    present = [str(path) for path in collisions if path.exists()]
    if present:
        raise FileExistsError(
            "refusing to overwrite artifacts from an existing training attempt: "
            + ", ".join(present)
        )


def _pool_rows(patch: Any, split: str) -> np.ndarray:
    raw = patch.split_pool[split]
    arrays: Iterable[Any] = raw.values() if isinstance(raw, dict) else (raw,)
    rows = [np.asarray(value, dtype=np.int64).reshape(-1) for value in arrays]
    return np.unique(np.concatenate(rows)) if rows else np.asarray([], dtype=np.int64)


def _save_patch_audits_exclusive(patch: Any, output_dir: Path) -> dict[str, Any]:
    """Persist the full fallback and partition provenance without touching geometry."""
    specifications = {
        "peer_selection_attempts.csv": (
            "peer_selection_attempts",
            {
                "split",
                "anchor_TIC",
                "spacing_tier",
                "outer_radius_cap",
                "expansion_used",
                "status",
                "relaxation_reason",
            },
        ),
        "peer_pool_assignment.csv": (
            "peer_pool_assignment",
            {"TIC", "split", "provenance", "hash_scheme"},
        ),
        "supplemental_ingestion_audit.csv": (
            "supplemental_ingestion_audit",
            {"source_row", "TIC", "status", "reason"},
        ),
    }
    summary: dict[str, Any] = {}
    for filename, (attribute, required_columns) in specifications.items():
        if not hasattr(patch, attribute):
            raise AttributeError(f"spread patch does not expose {attribute}")
        value = getattr(patch, attribute)
        frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{attribute} is missing audit columns: {missing}")
        path = output_dir / filename
        with path.open("x", newline="") as handle:
            frame.to_csv(handle, index=False)
        summary[attribute] = {
            "rows": int(len(frame)),
            "sha256": _sha256(path),
            "path": str(path.resolve()),
        }
    attempts = getattr(patch, "peer_selection_attempts")
    terminal = (
        attempts.sort_values(
            ["split", "anchor_TIC", "spacing_fallback_level"], kind="stable"
        )
        .groupby(["split", "anchor_TIC"], sort=False)
        .tail(1)
    )
    if len(terminal) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise AssertionError(
            "fallback audit does not contain one terminal attempt per anchor"
        )
    summary["terminal_attempt_spacing_tier_counts"] = {
        str(int(tier)): int(count)
        for tier, count in terminal["spacing_tier"].value_counts().sort_index().items()
    }
    summary["terminal_attempt_expansion_count"] = int(terminal["expansion_used"].sum())
    return summary


def _verify_patch_contract(
    patch: Any,
    selection: pd.DataFrame,
    reference_split_path: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if tuple(int(value) for value in patch.target) != (1, 4, 2):
        raise AssertionError(f"patch target changed: {patch.target}")
    counts = {split: len(patch.split_anchors[split]) for split in SPLITS}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"patch does not retain 320/40/40 anchors: {counts}")

    with reference_split_path.open() as handle:
        reference = json.load(handle)
    split_payload: dict[str, list[str]] = {}
    for split in SPLITS:
        reference_tics = [str(value) for value in reference[split]]
        anchor_rows = np.asarray(patch.split_anchors[split], dtype=np.int64)
        actual_tics = [str(value) for value in np.asarray(patch.tic)[anchor_rows]]
        if len(reference_tics) != EXPECTED_SPLIT_COUNTS[split]:
            raise AssertionError(
                f"reference {split} split no longer has the fixed size"
            )
        if set(actual_tics) != set(reference_tics):
            missing = sorted(set(reference_tics) - set(actual_tics))[:5]
            added = sorted(set(actual_tics) - set(reference_tics))[:5]
            raise AssertionError(
                f"{split} anchor TIC membership changed; missing={missing}, added={added}"
            )
        split_payload[split] = reference_tics

    pool_tics: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = _pool_rows(patch, split)
        pool_tics[split] = set(str(value) for value in np.asarray(patch.tic)[rows])
    overlaps = {
        "train_val": sorted(pool_tics["train"] & pool_tics["val"]),
        "train_test": sorted(pool_tics["train"] & pool_tics["test"]),
        "val_test": sorted(pool_tics["val"] & pool_tics["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(
            "peer TIC pools are not disjoint: "
            + str({name: values[:5] for name, values in overlaps.items() if values})
        )

    selection = selection.copy()
    selection["split"] = selection["split"].astype(str)
    selection["anchor_TIC"] = selection["anchor_TIC"].astype(str)
    selection["peer_TIC"] = selection["peer_TIC"].astype(str)
    selection["peer_order"] = selection["peer_order"].astype(int)
    aligned_groups = 0
    selected_peer_tics: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split in SPLITS:
        anchor_rows = np.asarray(patch.split_anchors[split], dtype=np.int64)
        peer_rows, peer_distances = patch.peers[split]
        if np.asarray(peer_rows).shape != (EXPECTED_SPLIT_COUNTS[split], 8):
            raise AssertionError(
                f"{split} peer table does not have eight peers per anchor"
            )
        for index, anchor_row in enumerate(anchor_rows):
            anchor_tic = str(patch.tic[int(anchor_row)])
            group = selection[
                (selection["split"] == split) & (selection["anchor_TIC"] == anchor_tic)
            ].sort_values("peer_order")
            if group["peer_order"].tolist() != list(range(1, 9)):
                raise AssertionError(
                    f"{split} TIC {anchor_tic}: audit slots are not 1..8"
                )
            actual_rows = np.asarray(peer_rows[index], dtype=np.int64)
            actual_tics = [str(value) for value in np.asarray(patch.tic)[actual_rows]]
            if actual_tics != group["peer_TIC"].tolist():
                raise AssertionError(
                    f"{split} TIC {anchor_tic}: live peers differ from audited peers"
                )
            if not np.allclose(
                np.asarray(peer_distances[index], dtype=float),
                group["anchor_distance"].to_numpy(dtype=float),
                atol=1e-4,
            ):
                raise AssertionError(
                    f"{split} TIC {anchor_tic}: live distances differ from the audit"
                )
            if len(set(actual_tics)) != 8 or anchor_tic in actual_tics:
                raise AssertionError(f"{split} TIC {anchor_tic}: TIC uniqueness failed")
            if not (
                (np.asarray(patch.sector)[actual_rows] == 1).all()
                and (np.asarray(patch.camera)[actual_rows] == 4).all()
                and (np.asarray(patch.ccd)[actual_rows] == 2).all()
            ):
                raise AssertionError(
                    f"{split} TIC {anchor_tic}: a peer left S1/C4/CCD2"
                )
            if not set(actual_tics).issubset(pool_tics[split]):
                raise AssertionError(
                    f"{split} TIC {anchor_tic}: peer came from another split"
                )
            selected_peer_tics[split].update(actual_tics)
            aligned_groups += 1

    selected_overlaps = {
        "train_val": selected_peer_tics["train"] & selected_peer_tics["val"],
        "train_test": selected_peer_tics["train"] & selected_peer_tics["test"],
        "val_test": selected_peer_tics["val"] & selected_peer_tics["test"],
    }
    if any(selected_overlaps.values()):
        raise AssertionError("selected peer TICs leak across splits")
    details = {
        "anchor_counts": counts,
        "audited_groups_aligned": aligned_groups,
        "peer_pool_unique_tics": {split: len(pool_tics[split]) for split in SPLITS},
        "selected_unique_peer_tics": {
            split: len(selected_peer_tics[split]) for split in SPLITS
        },
        "peer_pool_tic_disjoint": True,
        "selected_peer_tic_disjoint": True,
    }
    return split_payload, details


def _module_gradient_norm(module: torch.nn.Module) -> float:
    squared_norms = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norms.append(parameter.grad.detach().float().square().sum())
    if not squared_norms:
        return 0.0
    # Keep the reduction on the accelerator and synchronize only once per major
    # component.  Per-parameter host transfers make MPS training needlessly slow.
    total = torch.stack(squared_norms).sum().sqrt()
    return float(total.cpu())


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    result = {
        "physics_s4d": _module_gradient_norm(model.physics_encoder),
        "physics_projection": _module_gradient_norm(model.physics_projection),
        "physics_decoder": _module_gradient_norm(model.physics_decoder),
        "instrument_s4d": _module_gradient_norm(model.instrument_encoder),
        "instrument_projection": _module_gradient_norm(model.instrument_projection),
        "instrument_decoder": _module_gradient_norm(model.instrument_decoder),
    }
    result["physics_branch"] = math.sqrt(
        sum(
            result[name] ** 2
            for name in ("physics_s4d", "physics_projection", "physics_decoder")
        )
    )
    result["instrument_branch"] = math.sqrt(
        sum(
            result[name] ** 2
            for name in (
                "instrument_s4d",
                "instrument_projection",
                "instrument_decoder",
            )
        )
    )
    return result


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    hidden_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    raw = batch["anchor_raw"]
    valid = batch["anchor_valid_mask"].bool()
    masked, hidden, visible = mask_views(
        raw,
        valid,
        hidden_fraction=float(config["hidden_fraction"]),
        generator=generator,
        hidden_mask=hidden_mask,
    )
    tensors = (
        masked,
        visible,
        batch["peer_raw"],
        batch["peer_mask"].bool(),
        hidden,
        raw,
        valid,
    )
    masked, visible, peer_raw, peer_valid, hidden, raw, valid = (
        tensor.to(device) for tensor in tensors
    )
    outputs = model(masked, visible, peer_raw, peer_valid, hidden)
    hidden_valid = hidden & valid
    loss = masked_smooth_l1(outputs["reconstructed_raw"], raw, hidden_valid)
    outputs = model.apply_output_equations(raw, outputs)
    return loss, outputs, hidden_valid


def _masked_rms(values: torch.Tensor, valid: torch.Tensor) -> float:
    weights = valid.to(values.dtype)
    denominator = weights.sum().clamp(min=1)
    return float(
        torch.sqrt((values.square() * weights).sum() / denominator).detach().cpu()
    )


def _temporal_variation(values: torch.Tensor, valid: torch.Tensor) -> float:
    weights = valid.to(values.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp(min=1)
    means = (values * weights).sum(dim=1, keepdim=True) / denominator
    return _masked_rms(values - means, valid)


def _output_diagnostics(
    outputs: dict[str, torch.Tensor],
    raw: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    physics = outputs["physics_curve"].float()
    instrument = outputs["instrument_correction"].float()
    reconstructed = outputs["reconstructed_raw"].float()
    residual = raw.float() - reconstructed
    weights = valid.to(physics.dtype)
    dot = (physics * instrument * weights).sum()
    physics_norm = torch.sqrt((physics.square() * weights).sum()).clamp(min=1e-12)
    instrument_norm = torch.sqrt((instrument.square() * weights).sum()).clamp(min=1e-12)
    branch_cosine = dot / (physics_norm * instrument_norm)
    physics_rms = _masked_rms(physics, valid)
    instrument_rms = _masked_rms(instrument, valid)
    reconstructed_rms = _masked_rms(reconstructed, valid)
    return {
        "physics_output_rms": physics_rms,
        "instrument_output_rms": instrument_rms,
        "reconstructed_output_rms": reconstructed_rms,
        "residual_rms": _masked_rms(residual, valid),
        "physics_across_anchor_variation": float(
            physics.std(dim=0, unbiased=False).square().mean().sqrt().detach().cpu()
        ),
        "instrument_across_anchor_variation": float(
            instrument.std(dim=0, unbiased=False).square().mean().sqrt().detach().cpu()
        ),
        "physics_temporal_variation": _temporal_variation(physics, valid),
        "instrument_temporal_variation": _temporal_variation(instrument, valid),
        "branch_cosine": float(branch_cosine.detach().cpu()),
        "cancellation_ratio": reconstructed_rms
        / max(physics_rms + instrument_rms, 1e-12),
    }


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, float]:
    """One deterministic, cadence-weighted masked evaluation over a full split."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    numerator = 0.0
    denominator = 0
    stored: dict[str, list[torch.Tensor]] = {
        "physics_curve": [],
        "instrument_correction": [],
        "reconstructed_raw": [],
        "raw": [],
        "valid": [],
    }
    with torch.no_grad():
        for batch in loader:
            loss, outputs, loss_mask = _forward_batch(
                model, batch, config, device, generator=generator
            )
            count = int(loss_mask.sum().detach().cpu())
            numerator += float(loss.detach().cpu()) * count
            denominator += count
            for name in ("physics_curve", "instrument_correction", "reconstructed_raw"):
                stored[name].append(outputs[name].detach().float().cpu())
            stored["raw"].append(batch["anchor_raw"].detach().float().cpu())
            stored["valid"].append(batch["anchor_valid_mask"].detach().bool().cpu())
    if denominator == 0:
        raise RuntimeError("evaluation generated no hidden valid cadences")
    combined = {name: torch.cat(rows, dim=0) for name, rows in stored.items()}
    diagnostics = _output_diagnostics(
        {
            "physics_curve": combined["physics_curve"],
            "instrument_correction": combined["instrument_correction"],
            "reconstructed_raw": combined["reconstructed_raw"],
        },
        combined["raw"],
        combined["valid"],
    )
    diagnostics["masked_smooth_l1"] = numerator / denominator
    diagnostics["hidden_valid_cadences"] = denominator
    if was_training:
        model.train()
    return diagnostics


def _architecture_checks(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator().manual_seed(17042)
    raw = batch["anchor_raw"]
    valid = batch["anchor_valid_mask"].bool()
    masked, hidden, visible = mask_views(
        raw, valid, float(config["hidden_fraction"]), generator=generator
    )
    masked_d = masked.to(device)
    visible_d = visible.to(device)
    peers_d = batch["peer_raw"].to(device)
    peer_valid_d = batch["peer_mask"].bool().to(device)
    hidden_d = hidden.to(device)

    encoder_calls = 0
    projection_calls = 0

    def count_encoder(_module, _inputs, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    def count_projection(_module, _inputs, _output):
        nonlocal projection_calls
        projection_calls += 1

    encoder_hook = model.instrument_encoder.register_forward_hook(count_encoder)
    projection_hook = model.instrument_projection.register_forward_hook(
        count_projection
    )
    with torch.no_grad():
        outputs = model(masked_d, visible_d, peers_d, peer_valid_d, hidden_d)
    encoder_hook.remove()
    projection_hook.remove()

    batch_size = len(raw)
    expected_shapes = {
        "physics_latent": [batch_size, 64],
        "peer_tokens": [batch_size, 8, 4],
        "instrument_context": [batch_size, 32],
        "physics_curve": [batch_size, 1024],
        "instrument_correction": [batch_size, 1024],
        "reconstructed_raw": [batch_size, 1024],
    }
    actual_shapes = {name: list(outputs[name].shape) for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise AssertionError(
            f"architecture shapes differ: actual={actual_shapes}, expected={expected_shapes}"
        )
    if encoder_calls != 8 or projection_calls != 8:
        raise AssertionError(
            f"expected eight separate shared calls, got encoder={encoder_calls}, "
            f"projection={projection_calls}"
        )

    physics_ids = {id(parameter) for parameter in model.physics_decoder.parameters()}
    instrument_ids = {
        id(parameter) for parameter in model.instrument_decoder.parameters()
    }
    if physics_ids & instrument_ids:
        raise AssertionError("physics and instrument decoders share parameters")
    instrument_parameter_ids = [
        id(parameter) for parameter in model.instrument_encoder.parameters()
    ]
    if len(instrument_parameter_ids) != len(set(instrument_parameter_ids)):
        raise AssertionError("instrument encoder contains duplicate parameter objects")

    # Dynamic routing: changing one route's input must leave the other route bitwise
    # unchanged.  Conversely, each route must react to its own input.
    anchor_delta = torch.randn_like(masked_d) * visible_d.to(masked_d.dtype) * 0.1
    peer_delta = torch.randn_like(peers_d) * peer_valid_d.to(peers_d.dtype) * 0.1
    with torch.no_grad():
        changed_anchor = model(
            masked_d + anchor_delta, visible_d, peers_d, peer_valid_d, hidden_d
        )
        changed_peers = model(
            masked_d, visible_d, peers_d + peer_delta, peer_valid_d, hidden_d
        )
    if not torch.equal(
        outputs["instrument_correction"], changed_anchor["instrument_correction"]
    ):
        raise AssertionError("instrument route depends on anchor values")
    if not torch.equal(outputs["physics_curve"], changed_peers["physics_curve"]):
        raise AssertionError("physics route depends on peer values")
    if torch.equal(outputs["physics_curve"], changed_anchor["physics_curve"]):
        raise AssertionError("physics route does not respond to the anchor")
    if torch.equal(
        outputs["instrument_correction"], changed_peers["instrument_correction"]
    ):
        raise AssertionError("instrument route does not respond to peers")

    enriched = model.apply_output_equations(batch["anchor_raw"].to(device), outputs)
    if not torch.equal(
        enriched["reconstructed_raw"],
        enriched["physics_curve"] + enriched["instrument_correction"],
    ):
        raise AssertionError("reconstructed_raw equation changed")
    if not torch.equal(
        enriched["cleaned_curve"],
        batch["anchor_raw"].to(device) - enriched["instrument_correction"],
    ):
        raise AssertionError("cleaned_curve equation changed")
    if not torch.equal(
        enriched["residual"],
        batch["anchor_raw"].to(device) - enriched["reconstructed_raw"],
    ):
        raise AssertionError("residual equation changed")

    # A direct derivative check proves visible and invalid cadences have zero loss
    # weight, independently of the network's receptive field.
    synthetic_prediction = torch.zeros(2, 12, requires_grad=True)
    synthetic_target = torch.ones(2, 12)
    synthetic_valid = torch.tensor(
        [[1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]] * 2, dtype=torch.bool
    )
    synthetic_hidden = torch.tensor(
        [[0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]] * 2, dtype=torch.bool
    )
    synthetic_mask = synthetic_valid & synthetic_hidden
    synthetic_loss = masked_smooth_l1(
        synthetic_prediction, synthetic_target, synthetic_mask
    )
    synthetic_loss.backward()
    prediction_gradient = synthetic_prediction.grad
    if not bool((prediction_gradient[~synthetic_mask] == 0).all()):
        raise AssertionError("invalid or visible cadences contribute to the loss")
    if not bool((prediction_gradient[synthetic_mask] != 0).all()):
        raise AssertionError("a hidden valid cadence has zero direct loss gradient")
    perturbed_target = synthetic_target.clone()
    perturbed_target[~synthetic_mask] = 1e6
    perturbed_loss = masked_smooth_l1(
        synthetic_prediction.detach(), perturbed_target, synthetic_mask
    )
    if not torch.equal(synthetic_loss.detach(), perturbed_loss.detach()):
        raise AssertionError("values outside hidden-valid mask changed the loss")

    # Real backward pass: every requested learned component must receive a finite,
    # non-zero gradient from the single reconstruction loss.
    model.train()
    model.zero_grad(set_to_none=True)
    backward_generator = torch.Generator().manual_seed(18042)
    loss, _, loss_mask = _forward_batch(
        model, batch, config, device, generator=backward_generator
    )
    loss.backward()
    gradients = _gradient_norms(model)
    bad_gradients = {
        name: value
        for name, value in gradients.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if bad_gradients:
        raise AssertionError(f"missing/non-finite component gradients: {bad_gradients}")
    model.zero_grad(set_to_none=True)

    counts = model.parameter_count()
    if counts.get("total") != 1_777_764:
        raise AssertionError(f"unexpected model parameter count: {counts}")
    return {
        "status": "PASS",
        "shapes": actual_shapes,
        "parameters": counts,
        "routing": {
            "physics_input": "masked anchor and visible mask only",
            "instrument_input": "eight peer curves and peer validity only",
            "physics_unchanged_when_peers_change": True,
            "instrument_unchanged_when_anchor_changes": True,
            "physics_responds_to_anchor": True,
            "instrument_responds_to_peers": True,
        },
        "shared_instrument_route": {
            "instrument_encoder_instances": 1,
            "instrument_projection_instances": 1,
            "encoder_forward_calls_for_eight_peers": encoder_calls,
            "projection_forward_calls_for_eight_peers": projection_calls,
            "pooling": "none; [B,8,4] flattened in audited slot order",
        },
        "decoders_share_parameters": False,
        "output_equations_verified": True,
        "loss": {
            "name": "masked Smooth-L1",
            "prediction": "reconstructed_raw only",
            "mask": "hidden & valid anchor cadences",
            "visible_direct_gradient_zero": True,
            "invalid_direct_gradient_zero": True,
            "hidden_valid_cadences_in_real_batch": int(loss_mask.sum().detach().cpu()),
            "one_backward_loss": float(loss.detach().cpu()),
        },
        "gradient_norms_before_clipping": gradients,
    }


def _fixed_tiny_batch(dataset: CrossSectorAnchorDataset, count: int) -> dict[str, Any]:
    if len(dataset) < count:
        raise RuntimeError(
            f"tiny-overfit gate needs {count} anchors, got {len(dataset)}"
        )
    return default_collate([dataset[index] for index in range(count)])


def _tiny_overfit_gate(
    dataset: CrossSectorAnchorDataset,
    config: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    count = int(config["tiny_overfit_anchors"])
    batch = _fixed_tiny_batch(dataset, count)
    fixed_generator = torch.Generator().manual_seed(int(config["seed"]) + 42000)
    _, fixed_hidden, _ = mask_views(
        batch["anchor_raw"],
        batch["anchor_valid_mask"].bool(),
        float(config["hidden_fraction"]),
        generator=fixed_generator,
    )
    set_seed(int(config["seed"]) + 1)
    model = build_twodecoder_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    with torch.no_grad():
        initial, _, loss_mask = _forward_batch(
            model, batch, config, device, hidden_mask=fixed_hidden
        )
    initial_value = float(initial.detach().cpu())
    if not math.isfinite(initial_value) or initial_value <= 0:
        raise RuntimeError(f"tiny-overfit initial loss is invalid: {initial_value}")
    target_ratio = float(config["tiny_overfit_target_ratio"])
    minimum_steps = int(config["tiny_overfit_min_steps"])
    maximum_steps = int(config["tiny_overfit_max_steps"])
    best_value = initial_value
    trace = [{"step": 0, "loss": initial_value, "ratio": 1.0}]
    passed_at: int | None = None
    started = time.time()
    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss, _, _ = _forward_batch(
                model, batch, config, device, hidden_mask=fixed_hidden
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().cpu())
        best_value = min(best_value, value)
        if step == 1 or step % 10 == 0:
            trace.append({"step": step, "loss": value, "ratio": value / initial_value})
        if step >= minimum_steps and best_value / initial_value <= target_ratio:
            passed_at = step
            break
    _synchronize(device)
    result = {
        "status": "PASS" if passed_at is not None else "FAIL",
        "anchors": count,
        "fixed_mask": True,
        "hidden_valid_cadences": int(loss_mask.sum().detach().cpu()),
        "initial_loss": initial_value,
        "best_loss": best_value,
        "best_to_initial_ratio": best_value / initial_value,
        "required_ratio": target_ratio,
        "minimum_steps": minimum_steps,
        "maximum_steps": maximum_steps,
        "passed_at_step": passed_at,
        "seconds": time.time() - started,
        "trace": trace,
    }
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    if passed_at is None:
        raise RuntimeError(
            "tiny fixed-mask overfit gate failed: best/initial="
            f"{result['best_to_initial_ratio']:.4f}, required <= {target_ratio:.4f}"
        )
    return result


def _collapse_flags(
    metrics: dict[str, float], config: dict[str, Any]
) -> dict[str, bool]:
    rms_threshold = float(config["collapse_rms_threshold"])
    variation_threshold = float(config["collapse_variation_threshold"])
    return {
        "physics": bool(
            metrics["physics_output_rms"] <= rms_threshold
            or metrics["physics_across_anchor_variation"] <= variation_threshold
            or metrics["physics_temporal_variation"] <= variation_threshold
        ),
        "instrument": bool(
            metrics["instrument_output_rms"] <= rms_threshold
            or metrics["instrument_across_anchor_variation"] <= variation_threshold
            or metrics["instrument_temporal_variation"] <= variation_threshold
        ),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    validation_loss: float,
    geometry_summary_path: Path,
) -> None:
    checkpoint = {
        "format": "twodecoder_spread_v1",
        "model_class": "TwoDecoderSpreadModel",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "validation_masked_smooth_l1": float(validation_loss),
        "random_seed": int(config["seed"]),
        "architecture": {
            "physics_latent_dimension": 64,
            "instrument_token_dimension": 4,
            "number_of_peers": 8,
            "instrument_context_dimension": 32,
            "curve_length": 1024,
        },
        "peer_distance_rules": {
            "slots_1_to_4": "[128,384) pixels",
            "slots_5_to_8": "[384,768] pixels before outer-only expansion",
            "peer_spacing_tiers_pixels": [256, 192, 128],
        },
        "objective": "hidden-valid masked Smooth-L1 on reconstructed_raw only",
        "output_equations": {
            "reconstructed_raw": "physics_curve + instrument_correction",
            "cleaned_curve": "raw_anchor - instrument_correction",
            "residual": "raw_anchor - reconstructed_raw",
        },
        "scheduler": "constant learning rate; no scheduler state",
        "geometry_summary_sha256": _sha256(geometry_summary_path),
    }
    torch.save(checkpoint, path)


def _write_model_summary(
    path: Path,
    model: torch.nn.Module,
    *,
    best_epoch: int | None = None,
    best_validation: float | None = None,
) -> None:
    lines = [
        EXPECTED_RUN_NAME,
        "physics route: masked raw anchor [B,1024] -> independent physics S4D "
        "[B,32,16] -> flatten [B,512] -> Linear(512,64)",
        "physics decoder: Linear(64,256) -> GELU -> Linear(256,512) -> GELU "
        "-> Linear(512,1024)",
        "instrument route: each of 8 peers makes a separate call through one shared "
        "instrument S4D and one shared Linear(512,4)",
        "peer tokens: [B,8,4], no pooling; deterministic audited slot-order concat "
        "to instrument context [B,32]",
        "instrument decoder: independent Linear(32,256) -> GELU -> "
        "Linear(256,512) -> GELU -> Linear(512,1024)",
        "physics_curve = physics_decoder(physics_latent)",
        "instrument_correction = instrument_decoder(instrument_context)",
        "reconstructed_raw = physics_curve + instrument_correction",
        "cleaned_curve = raw_anchor - instrument_correction",
        "residual = raw_anchor - reconstructed_raw",
        "training loss: reconstructed_raw only, masked Smooth-L1 on hidden-valid "
        "anchor cadences",
        "No peer statistic, constructed correction/cleaned target, reference curve, "
        "or post-processing correction is used.",
        "Interpretation warning: reconstruction-only additive branches are not "
        "physically identified; names describe the constrained information routes, "
        "not proven physical components.",
        f"parameters: {model.parameter_count()}",
    ]
    if best_epoch is not None and best_validation is not None:
        lines.extend(
            (
                f"best epoch: {best_epoch}",
                f"best validation masked Smooth-L1: {best_validation:.9f}",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _plot_training_history(history_path: Path, output_path: Path) -> None:
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(history["epoch"], history["train_loss"], label="training loss")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="validation loss")
    axes[0, 0].set_ylabel("hidden-valid masked Smooth-L1")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(
        history["epoch"],
        history["train_physics_output_rms"],
        label="training physics-output RMS",
    )
    axes[0, 1].plot(
        history["epoch"],
        history["val_physics_output_rms"],
        label="validation physics-output RMS",
    )
    axes[0, 1].set_ylabel("physics output RMS")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(
        history["epoch"],
        history["train_instrument_output_rms"],
        label="training instrument-output RMS",
    )
    axes[1, 0].plot(
        history["epoch"],
        history["val_instrument_output_rms"],
        label="validation instrument-output RMS",
    )
    axes[1, 0].set_ylabel("instrument output RMS")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(
        history["epoch"],
        history["physics_branch_gradient_norm"],
        label="physics branch gradient",
    )
    axes[1, 1].plot(
        history["epoch"],
        history["instrument_branch_gradient_norm"],
        label="instrument branch gradient",
    )
    axes[1, 1].set_ylabel("pre-clip gradient norm")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.2)
    figure.suptitle("Two-decoder p64/i4 training diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def _complementary_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    patch: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    masks = complementary_masks(
        length=int(config["curve_length"]),
        n_masks=int(config["complementary_mask_count"]),
        block=int(config["complementary_mask_block"]),
    )
    stored: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "raw_anchor",
            "anchor_valid_mask",
            "peer_raw",
            "peer_valid_mask",
            "anchor_tic",
            "peer_tics",
            "anchor_row",
            "peer_rows",
            "peer_distances",
            "physics_curve",
            "instrument_correction",
            "reconstructed_raw",
            "cleaned_curve",
            "residual",
        )
    }
    for batch in loader:
        raw = batch["anchor_raw"].to(device)
        valid = batch["anchor_valid_mask"].bool().to(device)
        peer_raw = batch["peer_raw"].to(device)
        peer_valid = batch["peer_mask"].bool().to(device)
        _, peer_tokens, context = model.encode_instrument(peer_raw, peer_valid)
        if tuple(peer_tokens.shape[1:]) != (8, 4) or context.shape[1] != 32:
            raise AssertionError("instrument token shape changed during inference")
        instrument = model.instrument_decoder(context)
        physics = torch.zeros_like(raw)
        for mask in masks:
            cadence_mask = mask.to(device).unsqueeze(0).expand(len(raw), -1)
            hidden_valid = cadence_mask & valid
            visible = valid & ~cadence_mask
            masked = raw.masked_fill(hidden_valid, 0.0)
            _, latent = model.encode_physics(masked, visible)
            view = model.physics_decoder(latent)
            physics[:, cadence_mask[0]] = view[:, cadence_mask[0]]
        reconstructed = physics + instrument
        cleaned = raw - instrument
        residual = raw - reconstructed

        stored["raw_anchor"].append(raw.float().cpu().numpy())
        stored["anchor_valid_mask"].append(valid.cpu().numpy())
        stored["peer_raw"].append(peer_raw.float().cpu().numpy())
        stored["peer_valid_mask"].append(peer_valid.cpu().numpy())
        stored["anchor_tic"].append(batch["anchor_tic_ids"].cpu().numpy())
        stored["peer_tics"].append(batch["peer_tic_ids"].cpu().numpy())
        stored["anchor_row"].append(batch["anchor_row"].cpu().numpy())
        stored["peer_rows"].append(batch["peer_rows"].cpu().numpy())
        stored["peer_distances"].append(batch["peer_distances"].cpu().numpy())
        stored["physics_curve"].append(physics.float().cpu().numpy())
        stored["instrument_correction"].append(instrument.float().cpu().numpy())
        stored["reconstructed_raw"].append(reconstructed.float().cpu().numpy())
        stored["cleaned_curve"].append(cleaned.float().cpu().numpy())
        stored["residual"].append(residual.float().cpu().numpy())
    result = {name: np.concatenate(rows, axis=0) for name, rows in stored.items()}
    anchor_rows = result["anchor_row"].astype(np.int64)
    peer_rows = result["peer_rows"].astype(np.int64)
    result["anchor_detector_x"] = np.asarray(patch.det_x)[anchor_rows]
    result["anchor_detector_y"] = np.asarray(patch.det_y)[anchor_rows]
    result["peer_detector_x"] = np.asarray(patch.det_x)[peer_rows]
    result["peer_detector_y"] = np.asarray(patch.det_y)[peer_rows]
    result["complementary_masks"] = masks.numpy()
    return result


def _valid_line(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, values, np.nan)


def _square_geometry_limits(
    x: np.ndarray, y: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    span = max(x_max - x_min, y_max - y_min, 1.0) * 1.15
    x_mid, y_mid = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    return (x_mid - span / 2.0, x_mid + span / 2.0), (
        y_mid - span / 2.0,
        y_mid + span / 2.0,
    )


def _plot_heldout(
    arrays: dict[str, np.ndarray],
    plot_dir: Path,
    count: int,
) -> pd.DataFrame:
    plot_dir.mkdir(parents=True, exist_ok=False)
    total = len(arrays["raw_anchor"])
    count = min(int(count), total)
    indices = np.unique(np.linspace(0, total - 1, count, dtype=int))
    if len(indices) < count:
        raise AssertionError("could not select the requested number of held-out plots")
    records: list[dict[str, Any]] = []
    cadence = np.arange(arrays["raw_anchor"].shape[1])
    for plot_index, index in enumerate(indices):
        valid = arrays["anchor_valid_mask"][index].astype(bool)
        figure, axes = plt.subplots(4, 2, figsize=(16, 14))
        axes = axes.reshape(-1)
        series = (
            ("Raw anchor", arrays["raw_anchor"][index], "black"),
            (
                "reconstructed_raw = physics_curve + instrument_correction",
                arrays["reconstructed_raw"][index],
                "tab:purple",
            ),
            (
                "physics_curve = physics_decoder(physics_latent)",
                arrays["physics_curve"][index],
                "tab:blue",
            ),
            (
                "instrument_correction = instrument_decoder(instrument_context)",
                arrays["instrument_correction"][index],
                "tab:orange",
            ),
            (
                "cleaned_curve = raw_anchor - instrument_correction",
                arrays["cleaned_curve"][index],
                "tab:green",
            ),
            (
                "residual = raw_anchor - reconstructed_raw",
                arrays["residual"][index],
                "tab:red",
            ),
        )
        for axis, (title, values, color) in zip(axes[:6], series):
            axis.plot(cadence, _valid_line(values, valid), color=color, linewidth=0.8)
            axis.set_title(title, fontsize=9)
            axis.set_xlim(0, len(cadence) - 1)
            axis.grid(True, alpha=0.15)
            axis.set_ylabel("normalized flux")
        peer_axis = axes[6]
        for peer_index in range(8):
            peer_valid = arrays["peer_valid_mask"][index, peer_index].astype(bool)
            peer_axis.plot(
                cadence,
                _valid_line(arrays["peer_raw"][index, peer_index], peer_valid),
                linewidth=0.7,
                alpha=0.85,
                label=(
                    f"peer {peer_index + 1} TIC "
                    f"{int(arrays['peer_tics'][index, peer_index])}"
                ),
            )
        peer_axis.set_title(
            "All eight raw peer curves (individual; no pooling)", fontsize=9
        )
        peer_axis.set_xlim(0, len(cadence) - 1)
        peer_axis.set_xlabel("cadence-grid index")
        peer_axis.set_ylabel("normalized flux")
        peer_axis.legend(ncol=2, fontsize=6)
        peer_axis.grid(True, alpha=0.15)

        geometry = axes[7]
        anchor_x = float(arrays["anchor_detector_x"][index])
        anchor_y = float(arrays["anchor_detector_y"][index])
        peer_x = arrays["peer_detector_x"][index]
        peer_y = arrays["peer_detector_y"][index]
        geometry.scatter(
            anchor_x, anchor_y, marker="*", s=180, color="black", label="anchor"
        )
        colors = ["tab:blue"] * 4 + ["tab:orange"] * 4
        for peer_index, (x_value, y_value) in enumerate(zip(peer_x, peer_y)):
            geometry.plot(
                [anchor_x, x_value],
                [anchor_y, y_value],
                color=colors[peer_index],
                alpha=0.3,
                linewidth=0.8,
            )
            geometry.scatter(x_value, y_value, s=45, color=colors[peer_index])
            geometry.annotate(
                str(peer_index + 1),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        x_limits, y_limits = _square_geometry_limits(
            np.concatenate([[anchor_x], peer_x]), np.concatenate([[anchor_y], peer_y])
        )
        geometry.set_xlim(*x_limits)
        geometry.set_ylim(*y_limits)
        geometry.set_aspect("equal", adjustable="box")
        geometry.set_xlabel("detector X (pixels)")
        geometry.set_ylabel("detector Y (pixels)")
        geometry.set_title("Detector geometry (true 1:1 pixel scale)", fontsize=9)
        geometry.legend(fontsize=7)
        geometry.grid(True, alpha=0.2)
        figure.suptitle(
            f"Held-out TIC {int(arrays['anchor_tic'][index])} — learned additive outputs",
            fontsize=13,
        )
        figure.tight_layout()
        path = plot_dir / f"{plot_index:02d}_TIC{int(arrays['anchor_tic'][index])}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {
                "plot_index": plot_index,
                "test_index": int(index),
                "anchor_TIC": str(int(arrays["anchor_tic"][index])),
                "path": str(path.resolve()),
                "minimum_anchor_peer_distance": float(
                    np.min(arrays["peer_distances"][index])
                ),
                "maximum_anchor_peer_distance": float(
                    np.max(arrays["peer_distances"][index])
                ),
                "geometry_equal_scale": True,
            }
        )
    return pd.DataFrame(records)


def _per_star_metrics(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    valid = arrays["anchor_valid_mask"].astype(bool)
    records = []
    for index in range(len(valid)):
        mask = valid[index]

        def rms(values: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(values[index, mask]))))

        records.append(
            {
                "anchor_TIC": str(int(arrays["anchor_tic"][index])),
                "valid_cadences": int(mask.sum()),
                "physics_output_rms": rms(arrays["physics_curve"]),
                "instrument_output_rms": rms(arrays["instrument_correction"]),
                "reconstructed_output_rms": rms(arrays["reconstructed_raw"]),
                "residual_rms": rms(arrays["residual"]),
            }
        )
    return pd.DataFrame(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--supplemental-parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (HERE / "outputs" / EXPECTED_RUN_NAME).resolve()
    )
    expected_output = (HERE / "outputs" / EXPECTED_RUN_NAME).resolve()
    if output_dir != expected_output:
        raise ValueError(f"this isolated experiment must use {expected_output}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is intentionally first: no model is initialized and no training artifact
    # is written until the independent geometry feasibility audit has passed.
    geometry_summary, peer_selection = _verify_staged_geometry(output_dir)
    _refuse_training_collisions(output_dir)

    supplemental = (
        args.supplemental_parquet.expanduser().resolve()
        if args.supplemental_parquet is not None
        else _resolve_repo_path(config["supplemental_parquet"])
    )
    if not supplemental.is_file():
        raise FileNotFoundError(
            f"supplemental peer-only parquet is missing: {supplemental}"
        )
    reference_split = _resolve_repo_path(config["reference_split_tics"])
    if not reference_split.is_file():
        raise FileNotFoundError(f"reference anchor split is missing: {reference_split}")

    from disentangle_attempt.spread_peers import build_spread_patch_from_config

    print("geometry audit PASS; rebuilding the audited spread patch", flush=True)
    patch = build_spread_patch_from_config(
        config,
        supplemental_parquet=supplemental,
        verbose=True,
    )
    split_payload, patch_contract = _verify_patch_contract(
        patch, peer_selection, reference_split
    )
    patch_audits = _save_patch_audits_exclusive(patch, output_dir)
    datasets = {
        split: CrossSectorAnchorDataset(patch, split, seed=int(config["seed"]))
        for split in SPLITS
    }

    requested_device = args.device or config.get("device", "auto")
    set_seed(int(config["seed"]))
    device = pick_device(requested_device)
    use_amp = bool(config.get("mixed_precision")) and device.type == "cuda"
    workers = int(config.get("num_workers", 0))
    if workers and device.type == "mps":
        workers = 0
    print(
        f"device {device} | mixed precision {'on' if use_amp else 'off'} | "
        "single hidden-valid reconstruction loss",
        flush=True,
    )

    saved_config = dict(config)
    saved_config["supplemental_parquet"] = str(supplemental)
    saved_config["reference_split_tics"] = str(reference_split)
    saved_config["output_dir"] = str(output_dir)
    with (output_dir / "config.yaml").open("x") as handle:
        yaml.safe_dump(saved_config, handle, sort_keys=False)
    with (output_dir / "split_tics.json").open("x") as handle:
        json.dump(split_payload, handle, indent=2)
        handle.write("\n")

    audit_loader = DataLoader(
        datasets["train"],
        batch_size=int(config["anchors_per_step"]),
        shuffle=False,
        num_workers=workers,
    )
    audit_batch = next(iter(audit_loader))
    set_seed(int(config["seed"]))
    audit_model = build_twodecoder_model(config).to(device)
    architecture = _architecture_checks(audit_model, audit_batch, config, device)
    architecture["geometry_audit"] = {
        "status": geometry_summary["status"],
        "summary_sha256": _sha256(output_dir / "peer_geometry_summary.json"),
    }
    architecture["data_contract"] = patch_contract
    architecture["saved_patch_audits"] = patch_audits
    with (output_dir / "architecture_checks.json").open("x") as handle:
        json.dump(_jsonable(architecture), handle, indent=2)
        handle.write("\n")
    _write_model_summary(output_dir / "model_summary.txt", audit_model)
    del audit_model

    tiny = _tiny_overfit_gate(datasets["train"], config, device, use_amp)
    with (output_dir / "tiny_overfit.json").open("x") as handle:
        json.dump(_jsonable(tiny), handle, indent=2)
        handle.write("\n")
    print(
        f"tiny-overfit PASS at step {tiny['passed_at_step']}: "
        f"loss ratio {tiny['best_to_initial_ratio']:.4f}",
        flush=True,
    )

    # Reinitialize seed 42 after the deliberately separate tiny-overfit model.
    set_seed(int(config["seed"]))
    model = build_twodecoder_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler_generator = torch.Generator().manual_seed(int(config["seed"]) + 50000)
    batch_size = int(config["anchors_per_step"])
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        drop_last=True,
        num_workers=workers,
        sampler=RandomSampler(
            datasets["train"],
            replacement=True,
            num_samples=batch_size * int(config["max_train_steps_per_epoch"]),
            generator=sampler_generator,
        ),
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers
    )
    train_eval_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=False, num_workers=workers
    )

    history_path = output_dir / "training_history.csv"
    history_handle = history_path.open("x", newline="")
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
        mask_generator = torch.Generator().manual_seed(
            int(config["seed"]) * 1000 + epoch
        )
        loss_numerator = 0.0
        loss_denominator = 0
        gradient_sums = {
            name: 0.0
            for name in (
                "physics_branch",
                "instrument_branch",
                "physics_s4d",
                "physics_projection",
                "physics_decoder",
                "instrument_s4d",
                "instrument_projection",
                "instrument_decoder",
            )
        }
        train_physics_square_sum = torch.zeros((), device=device)
        train_instrument_square_sum = torch.zeros((), device=device)
        train_output_valid_count = torch.zeros((), device=device)
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
            output_valid = batch["anchor_valid_mask"].to(
                device=device, dtype=torch.float32
            )
            train_physics_square_sum += (
                outputs["physics_curve"].detach().float().square() * output_valid
            ).sum()
            train_instrument_square_sum += (
                outputs["instrument_correction"].detach().float().square()
                * output_valid
            ).sum()
            train_output_valid_count += output_valid.sum()
            steps += 1
        train_physics_rms = float(
            torch.sqrt(
                train_physics_square_sum / train_output_valid_count.clamp(min=1)
            ).cpu()
        )
        train_instrument_rms = float(
            torch.sqrt(
                train_instrument_square_sum / train_output_valid_count.clamp(min=1)
            ).cpu()
        )
        train_loss = loss_numerator / max(loss_denominator, 1)
        val = _evaluate_loader(
            model,
            val_loader,
            config,
            device,
            seed=int(config["seed"]) + 60000,
        )
        collapse_now = _collapse_flags(val, config)
        for branch in collapse_streak:
            collapse_streak[branch] = (
                collapse_streak[branch] + 1 if collapse_now[branch] else 0
            )
        _synchronize(device)
        epoch_seconds = time.time() - epoch_started
        wall_minutes = (time.time() - started) / 60.0
        gradient_means = {
            name: value / max(steps, 1) for name, value in gradient_sums.items()
        }
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["masked_smooth_l1"],
            "train_physics_output_rms": train_physics_rms,
            "train_instrument_output_rms": train_instrument_rms,
            "val_physics_output_rms": val["physics_output_rms"],
            "val_instrument_output_rms": val["instrument_output_rms"],
            "val_reconstructed_output_rms": val["reconstructed_output_rms"],
            "val_residual_rms": val["residual_rms"],
            "val_physics_across_anchor_variation": val[
                "physics_across_anchor_variation"
            ],
            "val_instrument_across_anchor_variation": val[
                "instrument_across_anchor_variation"
            ],
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
            "instrument_projection_gradient_norm": gradient_means[
                "instrument_projection"
            ],
            "instrument_decoder_gradient_norm": gradient_means["instrument_decoder"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "wall_minutes": wall_minutes,
            "physics_collapse_streak": collapse_streak["physics"],
            "instrument_collapse_streak": collapse_streak["instrument"],
        }
        writer.writerow(row)
        history_handle.flush()
        final_epoch = epoch
        print(
            f"epoch {epoch:02d} | train {train_loss:.6f} | "
            f"val {val['masked_smooth_l1']:.6f} | "
            f"physics RMS {val['physics_output_rms']:.6f} | "
            f"instrument RMS {val['instrument_output_rms']:.6f} | "
            f"grad physics {gradient_means['physics_branch']:.3e} | "
            f"grad instrument {gradient_means['instrument_branch']:.3e} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | {epoch_seconds:.1f}s",
            flush=True,
        )

        _save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            saved_config,
            epoch,
            val["masked_smooth_l1"],
            output_dir / "peer_geometry_summary.json",
        )
        if val["masked_smooth_l1"] < best["loss"]:
            best = {"loss": val["masked_smooth_l1"], "epoch": epoch}
            since_best = 0
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                saved_config,
                epoch,
                val["masked_smooth_l1"],
                output_dir / "peer_geometry_summary.json",
            )
        else:
            since_best += 1

        collapse_patience = int(config["collapse_patience_epochs"])
        collapsed = [
            name
            for name, streak in collapse_streak.items()
            if streak >= collapse_patience
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
            print("45-minute full-training limit reached", flush=True)
            break

    history_handle.close()
    training_seconds = time.time() - started
    _plot_training_history(history_path, output_dir / "training_curves.png")

    if best["epoch"] < 0 or not (output_dir / "best.pt").is_file():
        raise RuntimeError("training ended without a validation-selected checkpoint")
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != "twodecoder_spread_v1":
        raise AssertionError("best checkpoint format is not strict two-decoder v1")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    # The held-out test loader is scored exactly once, here, after the best epoch was
    # selected using validation only.
    final_train = _evaluate_loader(
        model, train_eval_loader, config, device, seed=int(config["seed"]) + 70001
    )
    final_val = _evaluate_loader(
        model, val_loader, config, device, seed=int(config["seed"]) + 70002
    )
    final_test = _evaluate_loader(
        model, test_loader, config, device, seed=int(config["seed"]) + 70003
    )
    heldout = _complementary_inference(model, test_loader, patch, config, device)
    np.savez_compressed(output_dir / "heldout_full_curve_arrays.npz", **heldout)
    per_star = _per_star_metrics(heldout)
    per_star.to_csv(output_dir / "heldout_per_star.csv", index=False)
    plot_manifest = _plot_heldout(
        heldout, output_dir / "plots", int(config["heldout_plot_count"])
    )
    if len(plot_manifest) < 12:
        raise AssertionError("fewer than 12 held-out plots were saved")
    plot_manifest.to_csv(output_dir / "plots" / "manifest.csv", index=False)

    final_collapse = _collapse_flags(final_val, config)
    any_collapse = any(final_collapse.values()) or stop_reason.startswith(
        "branch_collapse"
    )
    history = pd.read_csv(history_path)
    meaningful_gradients = bool(
        (history["physics_branch_gradient_norm"] > 0).all()
        and (history["instrument_branch_gradient_norm"] > 0).all()
    )
    metrics = {
        "run_name": EXPECTED_RUN_NAME,
        "status": "COMPLETE_WITH_COLLAPSE" if any_collapse else "COMPLETE",
        "device": str(device),
        "mixed_precision": use_amp,
        "geometry_feasible": geometry_summary["status"] == "PASS",
        "geometry": {
            "selected_anchor_counts": geometry_summary["selected_anchor_counts"],
            "fallback_counts": geometry_summary.get("final_selection_stage_counts"),
            "anchor_to_peer_distance_pixels": geometry_summary.get(
                "anchor_to_peer_distance_pixels"
            ),
            "peer_to_peer_distance_pixels": geometry_summary.get(
                "peer_to_peer_distance_pixels"
            ),
            "unique_selected_peer_TICs": geometry_summary.get(
                "unique_selected_peer_TICs"
            ),
            "fallback_and_partition_audits": patch_audits,
        },
        "parameters": model.parameter_count(),
        "best_epoch": int(best["epoch"]),
        "best_validation_masked_smooth_l1": float(best["loss"]),
        "final_full_split_evaluation": {
            "train": final_train,
            "validation": final_val,
            "test": final_test,
            "test_evaluation_count_after_checkpoint_selection": 1,
        },
        "training": {
            "last_epoch": final_epoch,
            "seconds": training_seconds,
            "minutes": training_seconds / 60.0,
            "stop_reason": stop_reason,
            "learning_rate_schedule": "constant",
            "meaningful_nonzero_gradients_both_branches": meaningful_gradients,
            "tiny_overfit": tiny,
        },
        "collapse_diagnostics": {
            "final_validation_flags": final_collapse,
            "sustained_collapse_stop": stop_reason.startswith("branch_collapse"),
            "either_output_branch_collapsed": any_collapse,
            "thresholds": {
                "rms": config["collapse_rms_threshold"],
                "variation": config["collapse_variation_threshold"],
                "patience_epochs": config["collapse_patience_epochs"],
            },
        },
        "heldout_full_curve_inference": {
            "method": "four deterministic complementary masks; physics predictions "
            "are retained only where hidden; instrument correction decoded once",
            "plot_count": int(len(plot_manifest)),
            "arrays": str((output_dir / "heldout_full_curve_arrays.npz").resolve()),
        },
        "output_contract": {
            "physics_curve": "physics_decoder(physics_latent)",
            "instrument_correction": "instrument_decoder(instrument_context)",
            "reconstructed_raw": "physics_curve + instrument_correction",
            "cleaned_curve": "raw_anchor - instrument_correction",
            "residual": "raw_anchor - reconstructed_raw",
            "only_training_prediction": "reconstructed_raw",
        },
        "identifiability": {
            "physically_proven_components": False,
            "statement": "These are learned additive components under asymmetric "
            "information routing. Reconstruction alone does not physically identify "
            "the decomposition; injection and peer-control tests remain future work.",
        },
        "ready_for_later_injection_and_peer_control_evaluation": bool(
            not any_collapse and meaningful_gradients
        ),
    }
    with (output_dir / "metrics.json").open("x") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)
        handle.write("\n")
    _write_model_summary(
        output_dir / "model_summary.txt",
        model,
        best_epoch=int(best["epoch"]),
        best_validation=float(best["loss"]),
    )

    print(
        f"complete: geometry PASS | best epoch {best['epoch']} | "
        f"train {final_train['masked_smooth_l1']:.6f} | "
        f"val {final_val['masked_smooth_l1']:.6f} | "
        f"test {final_test['masked_smooth_l1']:.6f} | "
        f"collapse {'yes' if any_collapse else 'no'} | "
        f"{training_seconds / 60.0:.1f} min",
        flush=True,
    )
    print(
        "The named branches are learned additive outputs, not yet physically proven "
        "components.",
        flush=True,
    )
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
