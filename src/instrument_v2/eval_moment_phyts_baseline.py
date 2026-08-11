"""MOMENT zero-shot vs the physics JEPA, on OUR cohort, split and probe.

The PhyTS paper reports MOMENT zero-shot at 0.8277 balanced accuracy, but that is an
EIGHT-class task on 25,935 mixed-cadence curves with their own split.  Our physics
numbers are SEVEN-class (INSTRUMENT/JUNK is absent from sector 14) on the 2,409-curve
PhyTS-matched S14 cohort.  Those two numbers are not comparable, and quoting them side
by side would be wrong.

This script removes that problem by changing exactly one thing: the encoder.  The
cohort, the TIC-disjoint split, the StandardScaler, the KNN(20) probe and the balanced
accuracy are all imported from ``eval_phyts_instrument_ab`` -- the same code that
produced the physics-JEPA numbers.  So the difference between arms is the
representation and nothing else.

    ARM=jepa   python -m src.instrument_v2.eval_moment_phyts_baseline
    ARM=moment python -m src.instrument_v2.eval_moment_phyts_baseline
    ARM=both   python -m src.instrument_v2.eval_moment_phyts_baseline   # default

Reference points on THIS cohort (7-class, chance 0.143): the ms16 physics JEPA scores
~0.629-0.636.  The 0.688 figure quoted elsewhere is from a different, larger eval and
is NOT the number to compare against here.

MOMENT needs ``pip install momentfm`` and downloads weights from HuggingFace, so the
first run must have outbound network (a login node).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.data.data import CLASSES, CLASS_TO_IDX
from src.instrument_v2.eval_phyts_instrument_ab import (  # exact reuse of the protocol
    DATA_PATH as PHYTS_PATH,
    DEVICE,
    GRID,
    assert_classes_present,
    classify,
    encode_physics,
    matched_split,
    physics_grid,
    _freeze,
)
from src.instrument_v2.eval_phyts_raw_tglc_ab import TGLC_PATH, match_phyts_tglc, quality_filter

ARM = os.environ.get("ARM", "both")
assert ARM in ("jepa", "moment", "both"), ARM
MOMENT_MODEL = os.environ.get("MOMENT_MODEL", "AutonLab/MOMENT-1-large")
PHYS_CKPT = os.environ.get(
    "JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth"
)
JEPA_READOUT = os.environ.get("JEPA_READOUT", "mean_std")
JEPA_NTOKENS = os.environ.get("JEPA_NTOKENS", "16")
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "moment_phyts_baseline")
)
BATCH = int(os.environ.get("BATCH", "32"))


def build_cohort():
    """The identical PhyTS-matched S14 cohort the physics A/B used."""
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next(
        (c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id") if c in phyts.columns),
        None,
    )
    if gaia_col is None:
        raise RuntimeError("PhyTS has no Gaia id column -- cannot match TGLC by GaiaDR3")
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})

    tglc_cols = set(pq.read_schema(TGLC_PATH).names)
    flux_col = "aperture_flux" if "aperture_flux" in tglc_cols else "flux"
    want = ["TIC", "sector", "GAIADR3", "time", flux_col, "TESS_flags", "TGLC_flags"]
    tglc = pd.read_parquet(TGLC_PATH, columns=want)
    tglc["TIC"] = tglc["TIC"].astype(str)
    tglc = tglc.rename(columns={flux_col: "aperture_flux"})

    matched, unmatched = match_phyts_tglc(phyts, tglc)
    print(
        f"PhyTS s14 rows {len(phyts)} | matched to raw TGLC {len(matched)} "
        f"| unmatched {len(unmatched)}",
        flush=True,
    )
    y = np.array([CLASS_TO_IDX[label] for label in matched["label"]], dtype=np.int64)
    tics = matched["TIC"].to_numpy().astype(str)
    return matched, tics, y


def build_grids(matched):
    """One quality-filtered, physics-normalized 1024 grid per star, shared by both arms."""
    n = len(matched)
    X = np.zeros((n, GRID), np.float32)
    M = np.zeros((n, GRID), np.float32)
    for i in range(n):
        time, flux = quality_filter(
            matched["time"].iloc[i],
            matched["aperture_flux"].iloc[i],
            matched["TESS_flags"].iloc[i],
            matched["TGLC_flags"].iloc[i],
        )
        X[i], M[i] = physics_grid(time, flux)
        if i % 400 == 0:
            print(f"  gridding {i}/{n}", flush=True)
    return X, M


def resample_for_moment(X: np.ndarray, M: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear-resample the 1024 grid to MOMENT's fixed context length.

    MOMENT-1 has a fixed 512-timestep context, so a 1024-point curve cannot be fed
    directly.  Resampling (rather than truncating) keeps the whole light curve, which
    matters because variability period is the discriminative feature here; truncating
    would throw away half the baseline and change what the task even is.
    """
    if X.shape[1] == target:
        return X, M
    source = np.linspace(0.0, 1.0, X.shape[1])
    grid = np.linspace(0.0, 1.0, target)
    Xr = np.stack([np.interp(grid, source, row) for row in X]).astype(np.float32)
    # A resampled point is valid only if it interpolates from observed neighbours.
    Mr = np.stack([np.interp(grid, source, row) for row in M]).astype(np.float32)
    Mr = (Mr > 0.5).astype(np.float32)
    return Xr, Mr


def encode_moment(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Frozen MOMENT embeddings, one vector per curve."""
    try:
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        raise SystemExit(
            "momentfm is not installed. On a login node:\n"
            "    pip install momentfm\n"
            f"(import failed: {exc})"
        )

    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL, model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model = _freeze(model.to(DEVICE).eval())

    seq_len = int(getattr(getattr(model, "config", object()), "seq_len", 512) or 512)
    Xr, Mr = resample_for_moment(X, M, seq_len)
    print(f"MOMENT {MOMENT_MODEL} | context {seq_len} | input {Xr.shape}", flush=True)

    outs = []
    with torch.no_grad():
        for start in range(0, len(Xr), BATCH):
            flux = torch.tensor(Xr[start : start + BATCH], dtype=torch.float32, device=DEVICE)
            mask = torch.tensor(Mr[start : start + BATCH], dtype=torch.float32, device=DEVICE)
            out = model(x_enc=flux.unsqueeze(1), input_mask=mask)  # (B, 1, L) univariate
            emb = getattr(out, "embeddings", out)
            outs.append(emb.reshape(emb.shape[0], -1).float().cpu().numpy())
            if start % (BATCH * 20) == 0:
                print(f"  MOMENT {start}/{len(Xr)}", flush=True)
    latents = np.concatenate(outs)
    if not np.isfinite(latents).all():
        raise RuntimeError("MOMENT produced non-finite embeddings")
    return latents


def encode_jepa(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Frozen physics JEPA embeddings, using the checkpoint's own readout settings."""
    os.environ["JEPA_READOUT"] = JEPA_READOUT
    os.environ["JEPA_NTOKENS"] = JEPA_NTOKENS
    from src.worked_folder.physics.latent_jepa import build_latent_jepa

    model = _freeze(build_latent_jepa().to(DEVICE))
    state = torch.load(PHYS_CKPT, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    _freeze(model)
    print(f"JEPA {PHYS_CKPT} | readout {JEPA_READOUT} | ntokens {JEPA_NTOKENS}", flush=True)
    return encode_physics(model, X, M)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    matched, tics, y = build_cohort()
    present = np.unique(y)
    print(
        f"cohort {len(y)} curves | {len(present)} classes present | chance "
        f"{1 / len(present):.3f}",
        flush=True,
    )
    X, M = build_grids(matched)

    # One split, shared by every arm -- the whole point of this script.
    train_idx, test_idx = matched_split(tics, y)
    assert_classes_present(y, train_idx, test_idx)
    split_hash = hashlib.sha256(
        "".join(f"{t}|{int(c)}\n" for t, c in zip(tics, y)).encode()
    ).hexdigest()

    arms = {}
    if ARM in ("jepa", "both"):
        arms["jepa"] = encode_jepa(X, M)
    if ARM in ("moment", "both"):
        arms["moment"] = encode_moment(X, M)

    results = {}
    for name, latents in arms.items():
        accuracy, recalls = classify(latents, y, train_idx, test_idx, present)
        results[name] = {
            "balanced_accuracy": accuracy,
            "latent_dim": int(latents.shape[1]),
            "per_class_recall": recalls,
        }
        print(f"\n{name}: balanced accuracy {accuracy:.4f} (dim {latents.shape[1]})", flush=True)
        for label, recall in sorted(recalls.items()):
            print(f"    {label:<14s} {recall:.3f}", flush=True)

    summary = {
        "cohort_curves": int(len(y)),
        "classes": [CLASSES[c] for c in present],
        "n_classes": int(len(present)),
        "chance": float(1 / len(present)),
        "split": "GroupShuffleSplit test_size=0.2 random_state=0, TIC-disjoint",
        "probe": "StandardScaler(train-fit) + KNeighborsClassifier(n_neighbors=20)",
        "metric": "balanced accuracy (unweighted mean per-class recall)",
        "cohort_hash": split_hash,
        "train_curves": int(len(train_idx)),
        "test_curves": int(len(test_idx)),
        "moment_model": MOMENT_MODEL if ARM in ("moment", "both") else None,
        "jepa_checkpoint": PHYS_CKPT if ARM in ("jepa", "both") else None,
        "arms": results,
        "comparability_note": (
            "7-class sector-14 cohort. NOT comparable to the PhyTS paper's 8-class "
            "0.8277 MOMENT / 0.887 S4D numbers, which use 25,935 mixed-cadence curves "
            "and a different split. Only the arms in this file share a protocol."
        ),
    }
    if len(results) == 2:
        summary["moment_minus_jepa"] = (
            results["moment"]["balanced_accuracy"] - results["jepa"]["balanced_accuracy"]
        )
        print(f"\nmoment - jepa = {summary['moment_minus_jepa']:+.4f}", flush=True)

    path = os.path.join(OUT_DIR, "summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"\nwrote {path}", flush=True)
    print(
        "Single split, single seed -- treat a small gap as provisional.",
        flush=True,
    )


if __name__ == "__main__":
    main()
