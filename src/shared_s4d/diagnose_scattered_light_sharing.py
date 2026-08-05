from __future__ import annotations
"""Diagnostic: are suspected scattered-light events genuinely SHARED between physically
nearby detector curves? No training, no model -- pure data analysis.

For ~100 strong localized anchor events (selected from the anchor curve ALONE, or from an
optional CSV), compare the anchor's event window against four groups at the SAME absolute
cadences: (1) its 15 nearest curves by DETECTOR_X/Y, (2) 15 random same-area curves,
(3) 15 distant same-CCD curves, (4) 15 curves from another CCD. Everything is masked and
locally detrended per-curve (never a group median). Writes a per-neighbor CSV, a JSON
group summary with bootstrap, overlay/heatmap/scatter/detector-map figures, and a verdict.

    S14_DATA=..._xy.parquet  python -m src.shared_s4d.diagnose_scattered_light_sharing
Env: S14_DATA, SPLIT_DIR, BASE_ART_DIR, EVENTS_CSV, N_EVENTS, N_STARS, ANCHOR_SEARCH,
     AMP_THRESH, OUT_DIR, SEED.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2_xy.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
EVENTS_CSV = os.environ.get("EVENTS_CSV", "")
N_EVENTS = int(os.environ.get("N_EVENTS", "100"))
N_STARS = int(os.environ.get("N_STARS", "1000"))
ANCHOR_SEARCH = int(os.environ.get("ANCHOR_SEARCH", "5000"))       # curves scanned for auto anchors
AMP_THRESH = float(os.environ.get("AMP_THRESH", "0.3"))            # |detrended amp| (norm units) = "event"
WINDOWS = (64, 128)
LAGS = (-2, -1, 0, 1, 2)
OUT_DIR = os.environ.get("OUT_DIR", os.path.join("artifacts", "shared_s4d", "scattered_light_sharing"))
SEED = int(os.environ.get("SEED", "0"))
GROUPS = ("nearest", "random", "distant", "diff_ccd")


# ---------- per-curve window helpers (mask-aware, never a group median) --------------------
def local_detrend(x, m, s, e, pad=64):
    """Detrended window x[s:e] with a local LINEAR baseline fit to the flanking valid points
    (the event core is excluded from the fit), estimated separately for THIS curve."""
    L = len(x); win = np.arange(s, e)
    flank = np.concatenate([np.arange(max(0, s - pad), s), np.arange(e, min(L, e + pad))])
    fv = flank[m[flank] > 0]
    if len(fv) >= 4:
        base = np.polyval(np.polyfit(fv, x[fv], 1), win)
    else:
        wv = win[m[win] > 0]
        base = np.full(len(win), np.median(x[wv]) if len(wv) else 0.0)
    return x[s:e] - base


def win_stats(da, db, va, vb):
    """(corr, cov, overlap) over shared valid cadences; NaN if <50% shared."""
    sh = va & vb; n = int(sh.sum())
    if n < 0.5 * len(da):
        return np.nan, np.nan, n
    a = da[sh] - da[sh].mean(); b = db[sh] - db[sh].mean()
    cov = float((a * b).mean())
    denom = float(np.sqrt((a * a).mean() * (b * b).mean()))
    return (float((a * b).mean() / denom) if denom > 1e-12 else np.nan), cov, n


def max_shift_corr(xa, ma, xb, mb, s, e):
    """Max zero-to-±2-lag correlation of the (detrended) anchor vs comparison event window."""
    da = local_detrend(xa, ma, s, e); va = ma[s:e] > 0
    best_c, best_lag, L = np.nan, 0, len(xb)
    for lag in LAGS:
        ss, ee = s + lag, e + lag
        if ss < 0 or ee > L:
            continue
        c, _, _ = win_stats(da, local_detrend(xb, mb, ss, ee), va, mb[ss:ee] > 0)
        if np.isfinite(c) and (not np.isfinite(best_c) or c > best_c):
            best_c, best_lag = c, lag
    return best_c, best_lag


def event_amp(x, m, s, e):
    """Signed event amplitude (mean of the locally detrended window over valid cadences)."""
    d = local_detrend(x, m, s, e); v = m[s:e] > 0
    return float(np.mean(d[v])) if v.sum() >= 0.5 * (e - s) else np.nan


# ---------- data ---------------------------------------------------------------------------
def load():
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    if not {"DETECTOR_X", "DETECTOR_Y"} <= set(df.columns):
        raise SystemExit("parquet lacks DETECTOR_X/DETECTOR_Y -- run merge_detector_positions.py")
    df = df[np.isfinite(df["DETECTOR_X"]) & np.isfinite(df["DETECTOR_Y"])]
    df = ensure_area_column(df)
    tr, va, te = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, tr)
    base = Sector14GroupStatDataset(df, va, t_range, "area", 32, min_valid=16)   # validation curves
    tics = np.asarray(base.tics).astype(str)
    meta = df.set_index(df["TIC"].astype(str))
    xy = meta.loc[tics, ["DETECTOR_X", "DETECTOR_Y"]].to_numpy(float)
    cam = meta.loc[tics, "camera"].to_numpy(int); ccd = meta.loc[tics, "ccd"].to_numpy(int)
    areas = np.asarray(base.areas)
    # cap the candidate pool at N_STARS per area (deterministic)
    rng = np.random.default_rng(SEED); keep = np.zeros(len(tics), bool)
    for a in np.unique(areas):
        idx = np.where(areas == a)[0]
        sel = idx if len(idx) <= N_STARS else rng.choice(idx, N_STARS, replace=False)
        keep[sel] = True
    return dict(X=base.X, M=base.M, tics=tics, areas=areas, xy=xy, cam=cam, ccd=ccd, pool=keep)


def select_anchors(d):
    X, M, pool = d["X"], d["M"], d["pool"]
    if EVENTS_CSV:
        ev = pd.read_csv(EVENTS_CSV); ev["TIC"] = ev["TIC"].astype(str)
        tidx = {t: i for i, t in enumerate(d["tics"])}
        out = []
        for _, r in ev.iterrows():
            if str(r["TIC"]) in tidx:
                s, e = int(r["start_cadence"]), int(r["end_cadence"])
                out.append((tidx[str(r["TIC"])], s, e, e - s, np.nan))
        return out
    rng = np.random.default_rng(SEED)
    rows = np.where(pool)[0]
    if len(rows) > ANCHOR_SEARCH:
        rows = rng.choice(rows, ANCHOR_SEARCH, replace=False)
    cand = []
    for i in rows:                                             # score = peak local excursion (anchor ONLY)
        x, m = X[i], M[i]
        for W in WINDOWS:
            for s in range(0, len(x) - W + 1, W // 2):
                e = s + W; v = m[s:e] > 0
                if v.sum() < 0.5 * W:
                    continue
                d0 = local_detrend(x, m, s, e)
                cand.append((float(np.max(np.abs(d0[v]))), i, s, e, W))
    cand.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for score, i, s, e, W in cand:
        if i in seen:
            continue
        seen.add(i); out.append((i, s, e, W, score))
        if len(out) >= N_EVENTS:
            break
    return out


# ---------- comparison groups --------------------------------------------------------------
def groups_for(anchor, d, rng):
    i = anchor; a = d["areas"][i]; c, cc = d["cam"][i], d["ccd"][i]
    pool = d["pool"]
    same_area = np.where(pool & (d["areas"] == a))[0]; same_area = same_area[same_area != i]
    same_ccd = np.where(pool & (d["cam"] == c) & (d["ccd"] == cc))[0]; same_ccd = same_ccd[same_ccd != i]
    other_ccd = np.where(pool & ~((d["cam"] == c) & (d["ccd"] == cc)))[0]
    dist = lambda rows: np.sqrt(((d["xy"][rows] - d["xy"][i]) ** 2).sum(1))
    g = {}
    if len(same_area) >= 15:
        order = same_area[np.argsort(dist(same_area))]
        g["nearest"] = order[:15]
        g["random"] = rng.choice(same_area, 15, replace=False)
    if len(same_ccd) >= 15:
        g["distant"] = same_ccd[np.argsort(dist(same_ccd))][-15:]        # farthest on the same CCD
    if len(other_ccd) >= 15:
        g["diff_ccd"] = rng.choice(other_ccd, 15, replace=False)
    return g


def measure(anchor, s, e, d, rng):
    X, M = d["X"], d["M"]
    xa, ma = X[anchor], M[anchor]
    amp_a = event_amp(xa, ma, s, e); sign_a = np.sign(amp_a)
    da = local_detrend(xa, ma, s, e); va = ma[s:e] > 0
    rows = []
    for gtype, members in groups_for(anchor, d, rng).items():
        for j in members:
            db = local_detrend(X[j], M[j], s, e); vb = M[j][s:e] > 0
            corr, cov, ov = win_stats(da, db, va, vb)
            mc, lag = max_shift_corr(xa, ma, X[j], M[j], s, e)
            amp_j = event_amp(X[j], M[j], s, e)
            dist = float(np.sqrt(((d["xy"][j] - d["xy"][anchor]) ** 2).sum()))
            same_sign_event = bool(np.isfinite(amp_j) and abs(amp_j) > AMP_THRESH and np.sign(amp_j) == sign_a)
            rows.append(dict(anchor_tic=d["tics"][anchor], neighbor_tic=d["tics"][j], group=gtype,
                             distance=dist, overlap=ov, corr=corr, max_corr=mc, lag=lag, cov=cov,
                             amp=amp_j, sign=int(np.sign(amp_j)) if np.isfinite(amp_j) else 0,
                             same_sign_event=same_sign_event, anchor_amp=amp_a))
    return rows


# ---------- stats / plots ------------------------------------------------------------------
def _q(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a):
        return {"median": None, "q1": None, "q3": None, "n": 0}
    return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "n": int(len(a))}


def bootstrap_diff(near, ctrl, n=2000, seed=0):
    near = np.asarray(near, float); near = near[np.isfinite(near)]
    ctrl = np.asarray(ctrl, float); ctrl = ctrl[np.isfinite(ctrl)]
    if len(near) < 5 or len(ctrl) < 5:
        return None
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(near, len(near))) - np.median(rng.choice(ctrl, len(ctrl))) for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"median_diff": float(np.median(diffs)), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def make_plots(df_rows, anchors, d):
    os.makedirs(OUT_DIR, exist_ok=True)
    # similarity vs detector distance (nearest + distant + diff_ccd)
    fig, ax = plt.subplots(figsize=(7, 5))
    for gtype, col in [("nearest", "tab:red"), ("distant", "tab:blue"), ("diff_ccd", "0.6")]:
        sub = df_rows[df_rows.group == gtype]
        ax.scatter(sub.distance, sub.max_corr, s=6, alpha=0.3, color=col, label=gtype)
    ax.set_xlabel("detector distance (px)"); ax.set_ylabel("max-shift correlation")
    ax.legend(); ax.set_title("event-window similarity vs detector distance")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "similarity_vs_distance.png"), dpi=130); plt.close(fig)

    # per strongest anchors: overlay + heatmap + detector XY amplitude map
    strong = sorted(anchors, key=lambda t: -(t[4] if np.isfinite(t[4]) else 0))[:4]
    for k, (i, s, e, W, score) in enumerate(strong):
        rng = np.random.default_rng(SEED + k); g = groups_for(i, d, rng)
        near = g.get("nearest", np.array([], int))
        gg = np.arange(max(0, s - W), min(d["X"].shape[1], e + W))
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 3]})
        ax[0].axvspan(s, e, color="0.9")
        ax[0].plot(gg, d["X"][i][gg], "k", lw=1.5, label="anchor")
        for j in near:
            ob = d["M"][j][gg] > 0
            ax[0].plot(gg[ob], d["X"][j][gg][ob], lw=0.5, alpha=0.5)
        ax[0].set_title(f"anchor {d['tics'][i]}  window [{s},{e}]  score {score:.2f}"); ax[0].legend(fontsize=7)
        # heatmap of nearest event windows sorted by distance
        order = near[np.argsort(np.sqrt(((d["xy"][near] - d["xy"][i]) ** 2).sum(1)))]
        H = np.vstack([local_detrend(d["X"][j], d["M"][j], s, e) for j in order]) if len(order) else np.zeros((1, e - s))
        im = ax[1].imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax[1].set_ylabel("nearest, sorted by distance"); ax[1].set_xlabel("cadence in window")
        fig.colorbar(im, ax=ax[1]); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"anchor{k}_overlay_heatmap.png"), dpi=130); plt.close(fig)
        # detector XY map colored by event amplitude (this event)
        same_ccd = np.where(d["pool"] & (d["cam"] == d["cam"][i]) & (d["ccd"] == d["ccd"][i]))[0]
        amps = np.array([event_amp(d["X"][j], d["M"][j], s, e) for j in same_ccd])
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(d["xy"][same_ccd, 0], d["xy"][same_ccd, 1], c=amps, cmap="RdBu_r", vmin=-1, vmax=1, s=10)
        ax.scatter(d["xy"][i, 0], d["xy"][i, 1], marker="*", s=160, edgecolor="k", facecolor="none")
        fig.colorbar(sc, ax=ax); ax.set_title(f"cam{d['cam'][i]} ccd{d['ccd'][i]} event amplitude")
        ax.set_xlabel("DETECTOR_X"); ax.set_ylabel("DETECTOR_Y")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"anchor{k}_detector_map.png"), dpi=130); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load()
    anchors = select_anchors(d)
    print(f"{len(anchors)} anchor events; pool {int(d['pool'].sum())} curves", flush=True)
    rng = np.random.default_rng(SEED)
    all_rows = []
    share_counts = []                                          # per anchor: #nearest-15 with same-sign event
    for (i, s, e, W, score) in anchors:
        rows = measure(i, s, e, d, rng)
        all_rows.extend(rows)
        near = [r for r in rows if r["group"] == "nearest"]
        share_counts.append(sum(r["same_sign_event"] for r in near))
    df_rows = pd.DataFrame(all_rows)
    df_rows.to_csv(os.path.join(OUT_DIR, "pairs.csv"), index=False)

    summary = {"n_anchors": len(anchors), "amp_thresh": AMP_THRESH,
               "nearest15_share_count": _q(share_counts),
               "groups": {}, "bootstrap_vs_nearest": {}, "spatial": {}}
    for g in GROUPS:
        sub = df_rows[df_rows.group == g]
        summary["groups"][g] = {"max_corr": _q(sub.max_corr), "corr": _q(sub.corr),
                                "cov": _q(sub.cov), "same_sign_frac": float(sub.same_sign_event.mean())
                                if len(sub) else None, "n_pairs": int(len(sub))}
    near_mc = df_rows[df_rows.group == "nearest"].max_corr
    near_share = df_rows[df_rows.group == "nearest"].same_sign_event.astype(float)
    for g in ("random", "distant", "diff_ccd"):
        sub = df_rows[df_rows.group == g]
        summary["bootstrap_vs_nearest"][g] = {
            "max_corr": bootstrap_diff(near_mc, sub.max_corr, seed=SEED),
            "same_sign_frac": bootstrap_diff(near_share, sub.same_sign_event.astype(float), seed=SEED)}
    # spatial dependence: slope of similarity vs distance among nearest+distant same-CCD pairs
    sc = df_rows[df_rows.group.isin(["nearest", "distant"])].dropna(subset=["max_corr", "distance"])
    if len(sc) > 20:
        slope = float(np.polyfit(sc.distance, sc.max_corr, 1)[0])
        summary["spatial"] = {"slope_corr_per_px": slope, "decreases_with_distance": bool(slope < 0)}

    # ---- verdict (needs BOTH cadence alignment vs controls AND spatial dependence) ----
    b = summary["bootstrap_vs_nearest"]
    beats_controls = all(b.get(g, {}).get("max_corr", {}) and b[g]["max_corr"]["excludes_zero"]
                         and b[g]["max_corr"]["median_diff"] > 0 for g in ("random", "distant", "diff_ccd") if g in b)
    spatial_ok = summary["spatial"].get("decreases_with_distance", False)
    share_med = summary["nearest15_share_count"]["median"]
    if beats_controls and spatial_ok:
        verdict = "SUPPORTED: nearest curves share events more than controls AND similarity falls with distance."
    elif share_med is not None and share_med <= 4:
        verdict = ("SPARSE_SHARING: only ~%s of the nearest 15 carry each event -> a 16-curve averaged "
                   "loss will DILUTE it (use per-curve / subset-aware handling)." % share_med)
    else:
        verdict = "NOT_SUPPORTED: nearest curves are no more similar than controls; detector-neighbor training alone cannot find these events."
    summary["verdict"] = verdict
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    make_plots(df_rows, anchors, d)

    print(json.dumps(summary, indent=2, default=float), flush=True)
    print("\n=== VERDICT ===\n" + verdict, flush=True)
    print(f"wrote pairs.csv, summary.json, and figures to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
