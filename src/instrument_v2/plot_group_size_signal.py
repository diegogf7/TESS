# All this code is from Claude
"""Signal-vs-group-size diagnostic + plot: show that area identity is a
GROUP property, invisible in one star and emerging as stars are pooled.

For each group size K, draw many pairs of fingerprints and correlate them:
  same_area   two disjoint K-star groups from the SAME area
  cross_area  two K-star groups, same camera/CCD, DIFFERENT area
The gap (same - cross) is ~0 at K=1 (a single star can't identify its area)
and opens up as K grows (the shared instrument common mode survives averaging
while each star's own physics/noise washes out ~ 1/sqrt(K)).

Reuses the exact fingerprint machinery (median / log-MAD, mutual-cadence
masking) from the JEPA pipeline. TRAIN stars only; test TICs never loaded.

    python -m src.instrument_v2.plot_group_size_signal
Env: N_PAIRS, K_VALUES (comma sep), SEED, GSS_ART_DIR, S14_DATA, ...
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    area_to_chip,
    ensure_area_column,
    group_statistics,
)
from src.instrument_v2.diagnose_raw_area_commonmode import masked_correlation
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit

N_PAIRS = int(os.environ.get("N_PAIRS", "400"))
K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "1,2,4,8,16,32").split(",")]
SEED = int(os.environ.get("SEED", "0"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "2000"))
ART_DIR = os.environ.get(
    "GSS_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "group_size_signal"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))


def combined_r(X, M, rows_a, rows_b, min_valid):
    med_a, mad_a, valid_a, _ = group_statistics(X[rows_a], M[rows_a], min_valid)
    med_b, mad_b, valid_b, _ = group_statistics(X[rows_b], M[rows_b], min_valid)
    mutual = (valid_a > 0) & (valid_b > 0)
    parts = [r for r in (masked_correlation(med_a, med_b, mutual),
                         masked_correlation(mad_a, mad_b, mutual))
             if np.isfinite(r)]
    return float(np.mean(parts)) if parts else float("nan")


def bootstrap_mean(values, rng):
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean()
             for _ in range(N_BOOTSTRAP)]
    return float(values.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5))


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, _, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    # k=1 so every area with >=2 stars is retained; we resample per-K below.
    base = Sector14GroupStatDataset(df, train_tics, t_range, "area", k=1)
    assert not set(base.tics) & test_tics, "test TIC leaked into diagnostic"
    X, M, labels = base.X, base.M, base.group_labels

    area_rows = {int(a): np.flatnonzero(labels == a) for a in np.unique(labels)}
    chip_areas = {}
    for area in area_rows:
        chip_areas.setdefault(area_to_chip(area), []).append(area)

    rows = []
    for k in K_VALUES:
        min_valid = max(1, math.ceil(k / 2))
        same_areas = [a for a, r in area_rows.items() if len(r) >= 2 * k]
        cross_chips = [c for c, areas in chip_areas.items()
                       if sum(len(area_rows[a]) >= k for a in areas) >= 2]
        if not same_areas or not cross_chips:
            print(f"K={k}: not enough stars, skipped", flush=True)
            continue

        same_vals, cross_vals = [], []
        for _ in range(N_PAIRS):
            area = same_areas[rng.integers(len(same_areas))]
            pick = rng.choice(area_rows[area], size=2 * k, replace=False)
            r = combined_r(X, M, pick[:k], pick[k:], min_valid)
            if np.isfinite(r):
                same_vals.append(r)

            chip = cross_chips[rng.integers(len(cross_chips))]
            areas_c = [a for a in chip_areas[chip] if len(area_rows[a]) >= k]
            a1, a2 = rng.choice(len(areas_c), size=2, replace=False)
            r = combined_r(X, M,
                           rng.choice(area_rows[areas_c[a1]], k, replace=False),
                           rng.choice(area_rows[areas_c[a2]], k, replace=False),
                           min_valid)
            if np.isfinite(r):
                cross_vals.append(r)

        same_m, same_lo, same_hi = bootstrap_mean(same_vals, rng)
        cross_m, cross_lo, cross_hi = bootstrap_mean(cross_vals, rng)
        gap = np.asarray(same_vals[:min(len(same_vals), len(cross_vals))]) - \
            np.asarray(cross_vals[:min(len(same_vals), len(cross_vals))])
        gap_m, gap_lo, gap_hi = bootstrap_mean(gap, rng)
        rows.append({"k": k, "same_mean": same_m, "same_ci": [same_lo, same_hi],
                     "cross_mean": cross_m, "cross_ci": [cross_lo, cross_hi],
                     "gap_mean": gap_m, "gap_ci": [gap_lo, gap_hi],
                     "n_same": len(same_vals), "n_cross": len(cross_vals)})
        print(f"K={k:2d}: same={same_m:.3f} cross={cross_m:.3f} "
              f"gap={gap_m:+.3f} [{gap_lo:+.3f},{gap_hi:+.3f}]", flush=True)

    summary = {"git_commit": git_commit(), "n_pairs": N_PAIRS, "rows": rows,
               "test_untouched": "TRAIN stars only; test TICs never loaded."}
    with open(os.path.join(ART_DIR, "group_size_signal.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); JSON written, skipping plot",
              flush=True)
        return

    ks = [r["k"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    def band(ax, key, color, label):
        m = [r[f"{key}_mean"] for r in rows]
        lo = [r[f"{key}_ci"][0] for r in rows]
        hi = [r[f"{key}_ci"][1] for r in rows]
        ax.plot(ks, m, "-o", color=color, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18)

    band(ax1, "same", "#2a6f97", "same area")
    band(ax1, "cross", "#c1121f", "same chip, different area")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ks)
    ax1.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("group size K (stars per fingerprint)")
    ax1.set_ylabel("fingerprint correlation")
    ax1.set_title("Same-area vs different-area coherence")
    ax1.legend(frameon=False)

    gap_m = [r["gap_mean"] for r in rows]
    gap_lo = [r["gap_ci"][0] for r in rows]
    gap_hi = [r["gap_ci"][1] for r in rows]
    ax2.axhline(0, color="0.6", lw=1, ls="--")
    ax2.plot(ks, gap_m, "-o", color="#333333")
    ax2.fill_between(ks, gap_lo, gap_hi, color="#333333", alpha=0.18)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ks)
    ax2.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("group size K (stars per fingerprint)")
    ax2.set_ylabel("same - cross  (area signal)")
    ax2.set_title("Area signal vs group size\n(~0 at K=1, opens with pooling)")

    fig.tight_layout()
    out = os.path.join(ART_DIR, "group_size_signal.png")
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
