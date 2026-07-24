from __future__ import annotations
"""Post-processing cleaner for the frozen group-32 DIRECT instrument decoder.

Nothing is retrained. For each VALIDATION star:
  quality-mask (TESS/TGLC flags) -> frozen S4D + predictor + DIRECT decoder
  -> 1024-cadence instrument TEMPLATE -> robust nonneg amplitude fit per
  observing segment -> instrument = a * template_centered -> cleaned = raw - instrument.

The decoder is used exactly as trained (one 1024-point curve); it is NOT changed
to predict CBV weights or components.

  python -m src.instrument_v2.scale_subtract_clean
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
)
from src.instrument_v2.decode_single_star_k8 import (
    GROUP_SIZE,
    CBV_RANK,
    MIN_VALID_STARS,
    DEVICE,
    build_decoder,
    decode,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import (
    ensure_splits,
    ensure_time_range,
    grid_curve_shared,
    shared_grid_bin,
)
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.train_sector14_jepa import git_commit

BAD_TESS_MASK = 1 | 4 | 16 | 32 | 16384          # == 16437
GRID = 1024
SEG_GAP = int(os.environ.get("SEG_GAP", "10"))    # >this many missing bins starts a new segment
MIN_SEG = int(os.environ.get("MIN_SEG", "8"))     # min valid points to fit an amplitude
SEED = int(os.environ.get("SEED", "0"))

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_v1"))
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_v1")
STAGE_B_CKPT = os.environ.get(
    "STAGE_B_CKPT", os.path.join(
        CKPT_DIR, f"group_cbv_mlp_g{GROUP_SIZE}_r{CBV_RANK}_mv{MIN_VALID_STARS}_s{SEED}_best.pth"))
DECODER_CKPT = os.environ.get(
    "DECODER_CKPT", os.path.join(GROUP_ART_DIR, "single_star_decode", "decoder.pth"))
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join(GROUP_ART_DIR, "scale_subtract_clean"))


# ------------------------------------------------------------------ quality
def cadence_good(tess, tglc):
    """Per-cadence keep mask: clean of the five bad TESS bits AND TGLC==0."""
    tess = np.asarray(tess, dtype=np.int64)
    tglc = np.asarray(tglc, dtype=np.int64)
    return ((tess & BAD_TESS_MASK) == 0) & (tglc == 0)


def prepare_good(time, flux, tess, tglc, t0, t1, grid_length=GRID):
    """Quality-filter BEFORE normalization/gridding. Only good (finite +
    flag-clean) cadences set the median/MAD and populate the shared grid;
    flagged / non-finite measurements never contribute to any bin.

    Returns (model_input[1024], valid_mask[1024] bool, median, mad).
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(time) & np.isfinite(flux) & cadence_good(tess, tglc)
    if not good.any():
        return np.zeros(grid_length), np.zeros(grid_length, dtype=bool), 0.0, 0.0
    normed, med, mad = normalize_median_mad(flux[good])                 # good-only stats
    X, M = grid_curve_shared(time[good], normed, t0, t1, grid_length)   # good-only grid
    return X.astype(np.float64), M > 0, float(med), float(mad)


# ---------------------------------------------------------------- amplitude
def robust_amplitude(raw_centered, dec_centered):
    """One nonnegative amplitude a >= 0 minimizing soft_l1(raw - a*dec)."""
    denom = float(np.dot(dec_centered, dec_centered))
    if denom <= 0:
        return 0.0
    a0 = max(0.0, float(np.dot(raw_centered, dec_centered) / denom))
    res = least_squares(lambda a: raw_centered - a[0] * dec_centered, x0=[a0],
                        bounds=(0.0, np.inf), loss="soft_l1")
    return float(res.x[0])


def segment_indices(valid_idx, gap=SEG_GAP, min_pts=MIN_SEG):
    """Split valid grid indices into continuous observing segments."""
    if len(valid_idx) == 0:
        return []
    splits = np.where(np.diff(valid_idx) > gap)[0] + 1
    return [s for s in np.split(valid_idx, splits) if len(s) >= min_pts]


def scale_and_subtract(raw, template, mask):
    """Per-segment robust fit -> instrument (centered) + cleaned (baseline kept)."""
    instrument = np.zeros(GRID, dtype=np.float64)
    amps = []
    for seg in segment_indices(np.flatnonzero(mask)):
        raw_c = raw[seg] - np.median(raw[seg])
        dec_c = template[seg] - np.median(template[seg])
        a = robust_amplitude(raw_c, dec_c)
        instrument[seg] = a * dec_c
        amps.append(a)
    cleaned = raw - instrument               # raw median preserved (instrument is centered)
    return instrument, cleaned, amps


def masked_corr(x, y, mask):
    x, y = x[mask], y[mask]
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rms(x, mask):
    x = x[mask]
    return float(np.sqrt(np.mean((x - np.median(x)) ** 2))) if x.size else np.nan


# --------------------------------------------------------------------- main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    if "TESS_flags" not in df.columns or "TGLC_flags" not in df.columns:
        raise SystemExit("FATAL: parquet lacks TESS_flags/TGLC_flags -- cannot "
                         "quality-filter; refusing to run without it")
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE,
                                      min_valid=MIN_VALID_STARS)
    assert not (set(val_ds.tics) & test_tics), "test TIC leaked into validation"

    flags = df.set_index(df["TIC"].astype(str))[["time", "flux", "TESS_flags", "TGLC_flags"]]

    # --- frozen model + existing trained DIRECT decoder ----------------------
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    decoder = build_decoder(1024).to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    decoder.eval()
    hashes = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor)}
    print(f"frozen Stage-B {STAGE_B_CKPT}\n decoder {DECODER_CKPT}", flush=True)

    corrs, rms_before, rms_after, fracs, all_amps, skipped = [], [], [], [], [], 0
    per_star = {}
    for i in range(len(val_ds.tics)):
        tic = str(val_ds.tics[i])
        rec = flags.loc[tic]
        if rec["TESS_flags"] is None or rec["TGLC_flags"] is None:
            skipped += 1
            continue
        t = np.asarray(rec["time"], dtype=np.float64)
        f = np.asarray(rec["flux"], dtype=np.float64)
        finite = np.isfinite(t) & np.isfinite(f)
        good = finite & cadence_good(rec["TESS_flags"], rec["TGLC_flags"])
        raw, mask, med, mad = prepare_good(rec["time"], rec["flux"],
                                           rec["TESS_flags"], rec["TGLC_flags"], *t_range)
        if mask.sum() < 64:                       # too few good grid bins -> skip
            skipped += 1
            continue
        template = decode(model, decoder, raw, mask.astype(np.float32)).astype(np.float64)
        instrument, cleaned, amps = scale_and_subtract(raw, template, mask)
        instrument[~mask] = np.nan                 # rejected bins: NaN, never exported
        cleaned[~mask] = np.nan

        corrs.append(masked_corr(raw, instrument, mask))
        rms_before.append(rms(raw, mask))
        rms_after.append(rms(cleaned, mask))
        fracs.append(float((finite.sum() - good.sum()) / max(1, finite.sum())))
        all_amps.extend(amps)
        per_star[tic] = {"area": int(val_ds.areas[i]), "amplitudes": amps,
                         "raw_vs_instrument_corr": corrs[-1],
                         "rms_before": rms_before[-1], "rms_after": rms_after[-1],
                         "frac_removed": fracs[-1]}

    assert state_hash(model.teacher) == hashes["teacher"], "teacher changed"
    assert state_hash(model.student) == hashes["student"], "student changed"
    assert state_hash(model.predictor) == hashes["predictor"], "predictor changed"

    def iqr(name, xs):
        a = np.asarray([v for v in xs if np.isfinite(v)], dtype=float)
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        print(f"{name}: median={med:.4f} IQR=[{q1:.4f},{q3:.4f}] (n={len(a)})", flush=True)
        return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": int(len(a))}

    # --- six-star diagnostic figure (2 rows shown per area, 6 columns) --------
    seen, picks = set(), []
    for i in range(len(val_ds.tics)):
        if str(val_ds.tics[i]) in per_star and int(val_ds.areas[i]) not in seen:
            seen.add(int(val_ds.areas[i])); picks.append(i)
        if len(picks) == 6:
            break
    x = np.arange(GRID)
    fig, axes = plt.subplots(len(picks), 6, figsize=(26, 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(picks):
        rec = flags.loc[str(val_ds.tics[i])]
        tt = np.asarray(rec["time"], dtype=np.float64)
        ff = np.asarray(rec["flux"], dtype=np.float64)
        raw, mask, med, mad = prepare_good(rec["time"], rec["flux"],
                                           rec["TESS_flags"], rec["TGLC_flags"], *t_range)
        template = decode(model, decoder, raw, mask.astype(np.float32)).astype(np.float64)
        instrument, cleaned, _ = scale_and_subtract(raw, template, mask)
        instrument[~mask] = np.nan
        cleaned[~mask] = np.nan
        bad = np.isfinite(tt) & np.isfinite(ff) & ~cadence_good(rec["TESS_flags"], rec["TGLC_flags"])
        sc = 1.4826 * mad if mad > 0 else 1.0
        bad_bins = shared_grid_bin(tt[bad], *t_range)
        bad_vals = (ff[bad] - med) / sc                # flagged points in the good-normalized frame

        def mk(a, m):
            out = np.full(GRID, np.nan); out[m] = a[m]; return out
        axes[row, 0].plot(x, mk(raw, mask), lw=0.6, color="0.4")
        axes[row, 1].plot(x, mk(template, mask), lw=0.7, color="tab:orange")
        axes[row, 2].plot(x, mk(instrument, mask), lw=0.7, color="tab:red")
        axes[row, 3].plot(x, mk(raw, mask), lw=0.6, color="0.4", label="raw")
        axes[row, 3].plot(x, mk(instrument, mask), lw=0.7, color="tab:red", label="instrument")
        axes[row, 4].plot(x, mk(cleaned, mask), lw=0.6, color="tab:green")
        axes[row, 5].plot(x, mk(raw, mask), lw=0.4, color="0.7")
        axes[row, 5].plot(bad_bins, bad_vals, ".", ms=3, color="tab:red")
        axes[row, 0].set_ylabel(f"area {int(val_ds.areas[i])}", fontsize=8)
    for c, t in enumerate(["raw", "direct decoder output", "scaled instrument",
                           "raw vs instrument", "cleaned (raw - instrument)",
                           "removed/flagged cadences"]):
        axes[0, c].set_title(t, fontsize=9)
    axes[0, 3].legend(fontsize=6, loc="upper right")
    fig.suptitle("scale-and-subtract cleaning (frozen group-32 DIRECT decoder)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT_DIR, "scale_subtract_clean.png"), dpi=130)
    plt.close(fig)

    report = {
        "git_commit": git_commit(), "stage_b_ckpt": STAGE_B_CKPT,
        "decoder_ckpt": DECODER_CKPT, "decoder_mode": "direct",
        "decoder_raw_output_dim": 1024, "quality_mask": BAD_TESS_MASK,
        "model_input": "good-cadence-only median/MAD normalized shared grid",
        "n_stars": len(corrs), "n_skipped_no_flags_or_under_64_bins": int(skipped),
        "fraction_cadences_removed": iqr("frac_removed", fracs),
        "fitted_amplitude": iqr("amplitude", all_amps),
        "raw_vs_instrument_corr": iqr("corr", corrs),
        "rms_before": iqr("rms_before", rms_before),
        "rms_after": iqr("rms_after", rms_after),
        "model_hashes_unchanged": True,
        "only_validation_used": True, "test_split_loaded": False,
        "per_star_first6": {str(val_ds.tics[i]): per_star[str(val_ds.tics[i])] for i in picks},
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_star_first6"}, indent=2),
          flush=True)
    print(f"wrote {OUT_DIR}/scale_subtract_clean.png + final_summary.json", flush=True)


if __name__ == "__main__":
    main()
