"""The single preprocessing function every curve goes through.

`preprocess_curve()` is the ONLY producer of model-visible flux in this package. The
anchor target, the masked current-sector physics input, the different-sector physics
input, the eight instrument peers and the quiet reference peers are all slices of the
same gridded arrays, so no branch can receive different filtering.

Policy: STRICT ZERO-FLAG. A cadence is valid only when

    np.isfinite(time) & np.isfinite(flux) & (TESS_flags == 0) & (TGLC_flags == 0)

Every nonzero TESS or TGLC flag removes the cadence -- momentum dumps, attitude
tweaks, Argabrightening, stray/scattered light, safe mode, coarse and Earth pointing,
calibration failures, cosmic-ray marks, manual exclusions, everything. Removed
cadences become missing gaps: `valid = 0`, and the zero sitting in the grid is never
an observation. They are excluded from the median/MAD, from the encoder's masked token
pooling, from the reconstruction loss and from every evaluation metric.

The original flag values are gridded and kept for auditing and reporting only. No
model branch ever sees the flux of a flagged cadence.

(This supersedes the source spec's "retain finite momentum-dump and scattered-light
cadences" instruction, at the user's explicit direction.)
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad

# TESS QUALITY bit values (TESS Archive Manual; same numbering as lightkurve's
# TessQualityFlags). Kept as named constants for the audit report only -- the validity
# rule is "any nonzero flag removes the cadence", so no bitmask is applied.
TESS_FLAG_NAMES = {
    1: "attitude tweak",
    2: "safe mode",
    4: "coarse point",
    8: "Earth point",
    16: "Argabrightening",
    32: "reaction-wheel desaturation (momentum dump)",
    64: "cosmic ray in optimal aperture",
    128: "manual exclude",
    256: "discontinuity corrected",
    512: "impulsive outlier",
    1024: "cosmic ray in collateral data",
    2048: "stray light",
    4096: "stray light (secondary)",
    8192: "planet search exclude",
    16384: "bad calibration exclude",
    32768: "insufficient targets for error correction",
}

MIN_VALID_CADENCES = 50

PreprocessedCurve = namedtuple(
    "PreprocessedCurve",
    ["curve", "valid", "flagged", "tess_flags", "tglc_flags", "n_valid"])


def preprocess_curve(cadence_num, time, flux, tess_flags, tglc_flags, cadence_grid):
    """One raw TGLC curve -> (normalized curve, valid mask, flagged-cadence mask).

    Placement onto `cadence_grid` is by EXACT cadence number: nothing is resampled,
    interpolated or smoothed. Flagged, non-finite and out-of-window cadences stay
    missing.

    `flagged` and the gridded `tess_flags`/`tglc_flags` are audit outputs; every
    cadence they mark has `valid = 0`, so none of their flux reaches a model branch.
    """
    length = len(cadence_grid)
    cadence = np.asarray(cadence_num, np.int64)
    flux = np.asarray(flux, np.float64)
    time = (np.full(len(cadence), np.nan) if time is None
            else np.asarray(time, np.float64))
    tess = (np.zeros(len(cadence), np.int64) if tess_flags is None
            else np.asarray(tess_flags, np.int64))
    tglc = (np.zeros(len(cadence), np.int64) if tglc_flags is None
            else np.asarray(tglc_flags, np.int64))
    if not (len(tess) == len(tglc) == len(flux) == len(time) == len(cadence)):
        raise ValueError("cadence/time/flux/TESS_flags/TGLC_flags lengths disagree")

    curve = np.zeros(length, dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    flagged = np.zeros(length, dtype=bool)
    tess_grid = np.zeros(length, dtype=np.int64)
    tglc_grid = np.zeros(length, dtype=np.int64)

    index = cadence - cadence_grid[0]
    inside = (index >= 0) & (index < length)
    # Audit arrays cover every in-window cadence, flagged or not.
    tess_grid[index[inside]] = tess[inside]
    tglc_grid[index[inside]] = tglc[inside]
    flagged[index[inside]] = (tess[inside] != 0) | (tglc[inside] != 0)

    keep = (inside & np.isfinite(time) & np.isfinite(flux)
            & (tess == 0) & (tglc == 0))
    if keep.sum() < 1:
        return PreprocessedCurve(curve, valid, flagged, tess_grid, tglc_grid, 0)

    # Median/MAD from the valid, unflagged cadences only -- the same ones the model
    # will see and the loss will score.
    normed, _, _ = normalize_median_mad(flux[keep])
    curve[index[keep]] = normed.astype(np.float32)
    valid[index[keep]] = True
    return PreprocessedCurve(curve, valid, flagged, tess_grid, tglc_grid,
                             int(keep.sum()))


def grid_values(cadence_num, values, cadence_grid, valid):
    """Place an auxiliary column (e.g. background) on the same grid and mask.

    Zeroed wherever `valid` is False, so a removed cadence contributes nothing.
    """
    length = len(cadence_grid)
    out = np.zeros(length, dtype=np.float32)
    if values is None or np.ndim(values) == 0:
        return out
    index = np.asarray(cadence_num, np.int64) - cadence_grid[0]
    inside = (index >= 0) & (index < length)
    out[index[inside]] = np.nan_to_num(
        np.asarray(values, np.float64)[inside]).astype(np.float32)
    return np.where(valid, out, 0.0).astype(np.float32)


def flag_removal_report(frame):
    """Cadences removed by each flag type, per sector. Bits overlap, so the per-bit
    rows sum to more than the total; `any flag` and `non-finite` are the real totals."""
    rows = []
    for sector, group in frame.groupby("sector"):
        tess = np.concatenate([np.asarray(a, np.int64) for a in group["TESS_flags"]])
        tglc = np.concatenate([np.asarray(a, np.int64) for a in group["TGLC_flags"]])
        flux = np.concatenate([np.asarray(a, np.float64) for a in group["flux"]])
        time = np.concatenate([np.asarray(a, np.float64) for a in group["time"]])
        total = len(tess)
        finite = np.isfinite(flux) & np.isfinite(time)

        def add(label, selected):
            rows.append({"sector": int(sector), "reason": label,
                         "cadences": int(selected.sum()), "total": total,
                         "percent": round(100.0 * selected.sum() / max(total, 1), 4)})

        add("non-finite time or flux", ~finite)
        for bit, name in sorted(TESS_FLAG_NAMES.items()):
            selected = (tess & bit).astype(bool)
            if selected.any():
                add(f"TESS bit {bit} ({name})", selected)
        if (tglc != 0).any():
            add("TGLC flag nonzero", tglc != 0)
        add("ANY flag or non-finite (total removed)",
            (tess != 0) | (tglc != 0) | ~finite)
        add("valid and retained", finite & (tess == 0) & (tglc == 0))
    return pd.DataFrame(rows)
