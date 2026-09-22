"""Three-panel before / subtraction / after figure for one TIC.

    python -m src.vicreg_jepa.plot_tic --tic 185336364 \
        --npz artifacts/vicreg_jepa/s14_curves.npz \
        --ckpt artifacts/vicreg_jepa/part1_s14/part1/part1_best.pt \
        --out plots/boyajian_cleaned.png --title "Boyajian's Star"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_common_mode import load_part1, common_mode_for_group
from .real_data import RealCurveSource, sphere_distance


def build(npz, ckpt, tic, split="val", group_size=32, device="cpu",
          peer_start=0, peer_seed=None):
    """peer_start skips the N nearest peers (so disjoint peer sets can be compared);
    peer_seed instead draws a random peer set from everything inside the area."""
    src = RealCurveSource(npz, split)
    hit = np.flatnonzero(src.tic == str(tic))
    if not len(hit):
        raise SystemExit(f"TIC {tic} is not in the {split!r} split of {npz}")
    j = int(hit[0])

    # nearest peers on the sky, from the SAME area, excluding the target
    same = np.flatnonzero((src.area == src.area[j]) & (np.arange(len(src.tic)) != j))
    if len(same) < group_size - 1:
        same = np.flatnonzero(np.arange(len(src.tic)) != j)
    d = sphere_distance(src.ra[same], src.dec[same], src.ra[j], src.dec[j])
    order = np.argsort(d)
    if peer_seed is not None:
        rng = np.random.default_rng(peer_seed)
        pool = order[:min(len(order), 400)]
        peers = same[rng.choice(pool, group_size - 1, replace=False)]
    else:
        sl = order[peer_start:peer_start + group_size - 1]
        if len(sl) < group_size - 1:
            raise SystemExit(f"only {len(order)} peers available; peer_start={peer_start} too large")
        peers = same[sl]
    grp = np.concatenate([[j], peers])

    f = src.flux[grp].astype(np.float32)
    o = src.observed[grp].astype(np.float32)
    enc, dec, blob = load_part1(ckpt, device)
    pred = common_mode_for_group(enc, dec, f, o, device)
    m = o[0] > 0
    dp = sphere_distance(src.ra[peers], src.dec[peers], src.ra[j], src.dec[j])
    return (np.where(m, f[0], np.nan), np.where(m, pred[0], np.nan),
            np.where(m, f[0] - pred[0], np.nan), src, j, blob,
            float(dp.max()) * 60)


def main(a):
    raw, sub, cln, src, j, blob, span = build(a.npz, a.ckpt, a.tic, a.split,
                                              a.group_size, "cpu",
                                              a.peer_start, a.peer_seed)
    t = np.arange(len(raw))
    lo, hi = np.nanpercentile(raw, [0.5, 99.5])
    pad = (hi - lo) * 0.5
    lim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, sharey=True)
    for axis, (y, c, title) in zip(ax, [
            (raw, "#33404a", "Original light curve"),
            (sub, "#c2410c", "Instrument signal removed"),
            (cln, "#1d6b52", "Cleaned light curve")]):
        axis.axhline(0, lw=.8, color="#d5dade", zorder=0)
        axis.plot(t, y, lw=1.2, color=c)
        axis.set_title(title, fontsize=14, loc="left", pad=8)
        axis.set_ylabel("Normalised flux", fontsize=12)
        axis.set_ylim(*lim)
        axis.tick_params(labelsize=11)
        axis.grid(axis="y", lw=.5, color="#eceff1", zorder=0)
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)
    ax[-1].set_xlabel("Cadence  (1024-bin sector grid)", fontsize=12)

    name = a.title or f"TIC {a.tic}"
    fig.suptitle(f"{name}  —  TIC {a.tic}, sector {int(src.sector[j])}, "
                 f"{a.group_size - 1} peers within {span:.1f}'", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    print(f"  raw std {np.nanstd(raw):.3f} -> cleaned std {np.nanstd(cln):.3f}")
    from .plot_common_mode import trend_power
    print(f"  trend power {trend_power(raw):.3f} -> {trend_power(cln):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", required=True)
    ap.add_argument("--npz", default="artifacts/vicreg_jepa/s14_curves.npz")
    ap.add_argument("--ckpt", default="artifacts/vicreg_jepa/part1_s14/part1/part1_best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--out", default="plots/tic_cleaned.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--peer-start", type=int, default=0)
    ap.add_argument("--peer-seed", type=int, default=None)
    main(ap.parse_args())
