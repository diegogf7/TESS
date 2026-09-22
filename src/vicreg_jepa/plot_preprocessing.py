"""Raw TGLC flux -> the array the model trains on. One row per star.

    python -m src.vicreg_jepa.plot_preprocessing --out plots/preprocessing.png
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .real_data import (BAD_TESS_MASK, quality_keep, normalize_median_mad,
                        build_shared_sector)


def main(parquet_glob, out_path, n=4, seed=3):
    import pyarrow.parquet as pq
    path = sorted(glob.glob(parquet_glob))[0]
    cols = ["time", "flux", "TIC", "TESS_flags", "TGLC_flags"]
    tbl = pq.read_table(path, columns=cols).to_pydict()
    rng = np.random.default_rng(seed)

    # prefer stars that actually have flagged cadences, so the step is visible
    cand = []
    for i in range(len(tbl["TIC"])):
        tf, gf = tbl["TESS_flags"][i], tbl["TGLC_flags"][i]
        if tf is None or gf is None:
            continue
        k = quality_keep(tf, gf, np.asarray(tbl["flux"][i], dtype=float))
        if (~k).mean() > 0.005:
            cand.append(i)
        if len(cand) > 400:
            break
    picks = rng.choice(cand, min(n, len(cand)), replace=False)

    fig, axes = plt.subplots(len(picks), 2, figsize=(14, 2.4 * len(picks)))
    fig.suptitle("Preprocessing: raw TGLC flux  ->  model input", fontsize=13, y=0.995)

    for row, i in enumerate(picks):
        t = np.asarray(tbl["time"][i], dtype=float)
        f = np.asarray(tbl["flux"][i], dtype=float)
        keep = quality_keep(tbl["TESS_flags"][i], tbl["TGLC_flags"][i], f)
        fz, med, mad = normalize_median_mad(f[keep])
        X, M, _ = build_shared_sector([t[keep]], [fz])
        x, m = X[0], M[0].astype(bool)

        a0, a1 = axes[row, 0], axes[row, 1]
        a0.plot(t[keep], f[keep], lw=0.6, color="#3b4a54", label="kept")
        a0.plot(t[~keep], f[~keep], ls="none", marker=".", ms=2.5,
                color="#c2410c", label=f"dropped ({100*(~keep).mean():.0f}%)")
        a0.set_ylabel(f"TIC {tbl['TIC'][i]}\nraw e-/s", fontsize=8)
        a0.legend(loc="best", fontsize=7.5, framealpha=0.9)
        lo, hi = np.percentile(f[keep], [0.5, 99.5])
        pad = (hi - lo) * 0.35 + 1e-9
        a0.set_ylim(lo - pad, hi + pad)

        y = np.where(m, x, np.nan)
        a1.axhline(0, lw=0.6, color="#b0b7bd", zorder=0)
        a1.plot(np.arange(len(y)), y, lw=0.7, color="#1d6b52")
        a1.set_ylabel("normalised flux", fontsize=8)
        lim = max(np.nanpercentile(np.abs(y), 98) * 1.4, 3.0)
        a1.set_ylim(-lim, lim)
        a1.text(0.012, 0.92, f"{int(m.sum())}/1024 bins observed",
                transform=a1.transAxes, fontsize=8, color="#1d6b52", va="top")

        if row == 0:
            a0.set_title("Before — raw flux, bad cadences marked", fontsize=10)
            a1.set_title("After — quality-masked, median/MAD normalised, 1024-bin grid",
                         fontsize=10)
        for a in (a0, a1):
            a.tick_params(labelsize=8)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)

    axes[-1, 0].set_xlabel("time (BTJD days)", fontsize=9)
    axes[-1, 1].set_xlabel("cadence (shared 1024-bin sector grid)", fontsize=9)
    fig.text(0.5, 0.005,
             f"Drop rule: (TESS_flags & {BAD_TESS_MASK}) != 0, or TGLC_flags != 0, or "
             "non-finite flux. Dropped cadences stay missing — never interpolated. "
             "Gaps become mask=0 and are excluded from every loss.",
             ha="center", fontsize=8.5, color="#5b6670")
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-glob",
                    default="disentangle_attempt/data_multichip/cross_sector_raw.parquet/*.parquet")
    ap.add_argument("--out", default="plots/preprocessing.png")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    main(args.parquet_glob, args.out, args.n)
