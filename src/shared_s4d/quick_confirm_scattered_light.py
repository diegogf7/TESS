from __future__ import annotations
"""Quick confirmation: are the strongest 'events' consistent with SCATTERED LIGHT, i.e.
does an independent BACKGROUND curve move with the stellar event at the same cadence?

Background here is TGLC's per-cadence `background` column (its estimate from NON-SOURCE
pixels around the star) -- NOT aperture/raw stellar flux. It is gridded exactly like the
flux (same quality mask, same shared grid). No model, no training. If the parquet has no
usable `background`, the script STOPS and says what FITS/FFI data is missing.

Reuses the diagnose_scattered_light_sharing anchor-selection + window metrics. Default
N_EVENTS=10 (the strongest events).

    S14_DATA=..._xy.parquet  python -m src.shared_s4d.quick_confirm_scattered_light
Env: S14_DATA, SPLIT_DIR, BASE_ART_DIR, N_EVENTS, N_STARS, ANCHOR_SEARCH, AMP_THRESH, OUT_DIR, SEED.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import (ensure_splits, ensure_time_range,
                                                grid_curve_shared, BAD_TESS_MASK)
from src.shared_s4d.diagnose_scattered_light_sharing import (
    local_detrend, win_stats, max_shift_corr, event_amp, groups_for, WINDOWS)

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2_xy.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
N_EVENTS = int(os.environ.get("N_EVENTS", "10"))
N_STARS = int(os.environ.get("N_STARS", "1000"))
ANCHOR_SEARCH = int(os.environ.get("ANCHOR_SEARCH", "5000"))
AMP_THRESH = float(os.environ.get("AMP_THRESH", "0.3"))
BG_CORR_MIN = float(os.environ.get("BG_CORR_MIN", "0.3"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join("artifacts", "shared_s4d", "scattered_light_confirmation"))
SEED = int(os.environ.get("SEED", "0"))
GRID = 1024


def _grid_series(df, col, t_range, normalize=False):
    """Grid a per-cadence column exactly like flux (same quality mask, same bins). Returns (V, MV)."""
    n = len(df); V = np.zeros((n, GRID), np.float32); MV = np.zeros((n, GRID), np.float32)
    for i in range(n):
        time = np.asarray(df["time"].iloc[i], float)
        val = np.asarray(df[col].iloc[i], float)
        tess = np.asarray(df["TESS_flags"].iloc[i], np.int64)
        tglc = np.asarray(df["TGLC_flags"].iloc[i], np.int64)
        if len(val) != len(time):
            continue
        good = np.isfinite(time) & np.isfinite(val) & ((tess & BAD_TESS_MASK) == 0) & (tglc == 0)
        if not good.any():
            continue
        v = val[good]
        if normalize:
            v = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9)
        V[i], MV[i] = grid_curve_shared(time[good], v, *t_range, GRID)
    return V, MV


def _grid_flag_frac(df, t_range, tess_bits=True):
    """Fraction of each grid bin's raw cadences that carried a flag (flagged cadences KEPT here)."""
    n = len(df); F = np.zeros((n, GRID), np.float32)
    for i in range(n):
        time = np.asarray(df["time"].iloc[i], float)
        tess = np.asarray(df["TESS_flags"].iloc[i], np.int64)
        tglc = np.asarray(df["TGLC_flags"].iloc[i], np.int64)
        flag = (((tess & BAD_TESS_MASK) != 0) if tess_bits else (tglc != 0)).astype(float)
        good = np.isfinite(time)
        if not good.any():
            continue
        F[i], _ = grid_curve_shared(time[good], flag[good], *t_range, GRID)
    return F


def load():
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    if not {"DETECTOR_X", "DETECTOR_Y"} <= set(df.columns):
        raise SystemExit("STOP: parquet lacks DETECTOR_X/DETECTOR_Y -- run merge_detector_positions.py")
    if "background" not in df.columns:
        raise SystemExit("STOP: parquet has NO 'background' column. The TGLC background (median of "
                         "non-source pixels) is unavailable, and stellar/aperture flux must NOT be "
                         "substituted. Re-extract from the TGLC FITS keeping the 'background' table "
                         "column (src/tglc/extract_raw_parquet_cadence.py), then rebuild the parquet.")
    df = df[np.isfinite(df["DETECTOR_X"]) & np.isfinite(df["DETECTOR_Y"])]
    df = ensure_area_column(df)
    tr, va, te = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, tr)
    base = Sector14GroupStatDataset(df, va, t_range, "area", 32, min_valid=16)
    tics = np.asarray(base.tics).astype(str)
    filt = df[df["sector"] == 14]
    filt = filt[filt["TIC"].astype(str).isin(set(va))].reset_index(drop=True)
    if not np.array_equal(filt["TIC"].astype(str).to_numpy(), tics):
        raise SystemExit("STOP: background row order != flux row order -- cannot align")
    if filt["background"].isna().mean() > 0.5:
        raise SystemExit("STOP: 'background' column is >50% null -- FITS background missing for most stars.")
    B, MB = _grid_series(filt, "background", t_range, normalize=True)
    Ftess = _grid_flag_frac(filt, t_range, tess_bits=True)
    Ftglc = _grid_flag_frac(filt, t_range, tess_bits=False)
    if float((MB > 0).mean()) < 0.05:
        raise SystemExit("STOP: gridded background is almost entirely empty -- unusable background data.")

    meta = df.set_index(df["TIC"].astype(str))
    xy = meta.loc[tics, ["DETECTOR_X", "DETECTOR_Y"]].to_numpy(float)
    cam = meta.loc[tics, "camera"].to_numpy(int); ccd = meta.loc[tics, "ccd"].to_numpy(int)
    areas = np.asarray(base.areas)
    rng = np.random.default_rng(SEED); keep = np.zeros(len(tics), bool)
    for a in np.unique(areas):
        idx = np.where(areas == a)[0]
        sel = idx if len(idx) <= N_STARS else rng.choice(idx, N_STARS, replace=False)
        keep[sel] = True
    return dict(X=base.X, M=base.M, B=B, MB=MB, Ftess=Ftess, Ftglc=Ftglc,
                tics=tics, areas=areas, xy=xy, cam=cam, ccd=ccd, pool=keep)


def top_anchors(d, n):
    """The n strongest localized anchor events (same logic/seed as the sharing diagnostic)."""
    rng = np.random.default_rng(SEED); rows = np.where(d["pool"])[0]
    if len(rows) > ANCHOR_SEARCH:
        rows = rng.choice(rows, ANCHOR_SEARCH, replace=False)
    cand = []
    for i in rows:
        x, m = d["X"][i], d["M"][i]
        for W in WINDOWS:
            for s in range(0, len(x) - W + 1, W // 2):
                e = s + W; v = m[s:e] > 0
                if v.sum() < 0.5 * W:
                    continue
                d0 = local_detrend(x, m, s, e)
                cand.append((float(np.max(np.abs(d0[v]))), i, s, e, W))
    cand.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for sc, i, s, e, W in cand:
        if i in seen:
            continue
        seen.add(i); out.append((i, s, e, W, sc));
        if len(out) >= n:
            break
    return out


def quiet_window(d, i, W):
    """A matched QUIET control window in the SAME curve (smallest local excursion)."""
    x, m = d["X"][i], d["M"][i]; best = None
    for s in range(0, len(x) - W + 1, W // 2):
        e = s + W; v = m[s:e] > 0
        if v.sum() < 0.5 * W:
            continue
        sc = float(np.max(np.abs(local_detrend(x, m, s, e)[v])))
        if best is None or sc < best[0]:
            best = (sc, s, e)
    return (best[1], best[2]) if best else (0, W)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load()
    anchors = top_anchors(d, N_EVENTS)
    print(f"{len(anchors)} strongest events; pool {int(d['pool'].sum())} curves", flush=True)
    rng = np.random.default_rng(SEED)
    rows, amp_star_all, amp_bg_all = [], [], []
    for k, (i, s, e, W, score) in enumerate(anchors):
        # stellar vs its own background at 0/±2 lag
        bg_corr, bg_lag = max_shift_corr(d["X"][i], d["M"][i], d["B"][i], d["MB"][i], s, e)
        amp_star = event_amp(d["X"][i], d["M"][i], s, e)
        amp_bg = event_amp(d["B"][i], d["MB"][i], s, e)
        # matched QUIET control window in the same curve
        qs, qe = quiet_window(d, i, W)
        ctrl_corr, _ = max_shift_corr(d["X"][i], d["M"][i], d["B"][i], d["MB"][i], qs, qe)
        # spatial coherence: do the nearest-15 backgrounds carry the same-sign event?
        g = groups_for(i, d, rng); near = g.get("nearest", np.array([], int))
        dist_grp = g.get("distant", np.array([], int))
        def bg_share(members):
            if not len(members):
                return np.nan
            sig = [event_amp(d["B"][j], d["MB"][j], s, e) for j in members]
            return float(np.mean([np.isfinite(a) and abs(a) > AMP_THRESH and np.sign(a) == np.sign(amp_bg)
                                  for a in sig]))
        near_bg_share, dist_bg_share = bg_share(near), bg_share(dist_grp)
        # flags during the event (reported, NOT used as proof)
        ft = float(np.mean(d["Ftess"][i][s:e])); fg = float(np.mean(d["Ftglc"][i][s:e]))
        rows.append(dict(anchor_tic=d["tics"][i], s=s, e=e, W=W, score=score,
                         stellar_bg_corr=bg_corr, bg_lag=bg_lag, control_bg_corr=ctrl_corr,
                         amp_star=amp_star, amp_bg=amp_bg, near_bg_share=near_bg_share,
                         dist_bg_share=dist_bg_share, flag_frac_tess=ft, flag_frac_tglc=fg))
        if np.isfinite(amp_star) and np.isfinite(amp_bg):
            amp_star_all.append(amp_star); amp_bg_all.append(amp_bg)
        _overlay(d, i, s, e, W, near, k)

    rf = pd.DataFrame(rows); rf.to_csv(os.path.join(OUT_DIR, "event_results.csv"), index=False)

    def med(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else None
    amp_s = np.asarray(amp_star_all, float); amp_b = np.asarray(amp_bg_all, float)
    amp_corr = float(np.corrcoef(amp_s, amp_b)[0, 1]) if len(amp_s) >= 3 else None
    # per-event coupling SIGN: sign(amp_star * amp_bg) should agree across >=80% of events
    prod = amp_s * amp_b; prod = prod[np.isfinite(prod) & (prod != 0)]
    if len(prod):
        pos_frac = float(np.mean(prod > 0)); dom = float(max(pos_frac, 1 - pos_frac))
        relationship = "positive" if pos_frac >= 0.5 else "negative"
    else:
        dom, relationship = 0.0, "undetermined"
    sign_consistent = dom >= 0.8

    c_same_cadence = med(rf.stellar_bg_corr) is not None and med(rf.stellar_bg_corr) > BG_CORR_MIN \
        and med(np.abs(rf.bg_lag)) <= 2
    # magnitude of coupling (either sign) >= 0.5 AND a consistent per-event sign
    c_amp_corr = amp_corr is not None and abs(amp_corr) >= 0.5 and sign_consistent
    c_vs_control = med(rf.stellar_bg_corr) is not None and med(rf.control_bg_corr) is not None \
        and med(rf.stellar_bg_corr) > med(rf.control_bg_corr) + 0.1
    c_spatial = med(rf.near_bg_share) is not None and med(rf.near_bg_share) > 0.5 \
        and (med(rf.dist_bg_share) is None or med(rf.near_bg_share) > med(rf.dist_bg_share))
    consistent = bool(c_same_cadence and c_amp_corr and c_vs_control and c_spatial)

    summary = {"n_events": len(anchors), "amp_thresh": AMP_THRESH,
               "median_stellar_bg_corr": med(rf.stellar_bg_corr),
               "median_control_bg_corr": med(rf.control_bg_corr),
               "median_abs_bg_lag": med(np.abs(rf.bg_lag)),
               "amp_star_vs_amp_bg_corr": amp_corr,
               "amp_abs_corr": abs(amp_corr) if amp_corr is not None else None,
               "amp_relationship": relationship, "amp_sign_consistency": dom,
               "median_nearest_bg_share": med(rf.near_bg_share),
               "median_distant_bg_share": med(rf.dist_bg_share),
               "median_flag_frac_tess": med(rf.flag_frac_tess),
               "median_flag_frac_tglc": med(rf.flag_frac_tglc),
               "criteria": {"background_same_cadence": bool(c_same_cadence),
                            "amp_correlates_with_background_abs>=0.5_signconsistent": bool(c_amp_corr),
                            "stronger_than_control": bool(c_vs_control),
                            "spatially_coherent": bool(c_spatial)},
               "consistent_with_background_systematic": consistent}
    if consistent:
        summary["verdict"] = (f"Consistent with a background-driven detector systematic, likely scattered "
                              f"light ({relationship} amplitude coupling, |r|={abs(amp_corr):.2f}, sign "
                              f"consistent in {dom * 100:.0f}% of events). NOT claimed as definitive "
                              f"scattered light: Earth/Moon geometry has not been checked.")
    else:
        summary["verdict"] = ("NOT a background-driven systematic by this test: " +
                              ", ".join(k for k, v in summary["criteria"].items() if not v) + " failed.")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float), flush=True)
    print("\n=== VERDICT ===\n" + summary["verdict"], flush=True)
    print(f"wrote event_results.csv, summary.json, {len(anchors)} overlay plots to {OUT_DIR}", flush=True)


def _overlay(d, i, s, e, W, near, k):
    gg = np.arange(max(0, s - W), min(GRID, e + W))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axvspan(s, e, color="0.92")
    ob = d["M"][i][gg] > 0
    ax.plot(gg[ob], d["X"][i][gg][ob], "k", lw=1.6, label="stellar flux (anchor)")
    if len(near):
        Mr = d["M"][near][:, gg] > 0; num = (d["X"][near][:, gg] * Mr).sum(0)
        den = Mr.sum(0); nm = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        ax.plot(gg, nm, color="tab:orange", lw=1.0, label="neighbor median (15 nearest)")
    ob2 = d["MB"][i][gg] > 0
    ax.plot(gg[ob2], d["B"][i][gg][ob2], color="tab:green", lw=1.2, label="background (non-source px)")
    ax.set_title(f"event {k}: anchor {d['tics'][i]}  window [{s},{e}]"); ax.legend(fontsize=7)
    ax.set_xlabel("cadence"); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"event{k}_overlay.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
