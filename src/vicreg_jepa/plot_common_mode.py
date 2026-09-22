"""Before / after figure: what Part 1 removes from a light curve.

Part 1 never sees the target star's own latent. It reconstructs each curve from
the mean of its 31 region peers' latents, so what it can predict is the shared
instrumental signal -- the common mode. Subtracting that prediction is the
"after" curve.

    python -m src.vicreg_jepa.plot_common_mode \
        --ckpt artifacts/vicreg_jepa/pilot_real/part1/part1_best.pt \
        --out plots/part1_common_mode.png
"""
import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .losses import leave_one_out_mean
from .models import S4Encoder, CommonModeDecoder
from .real_data import RealCurveSource


def load_part1(ckpt_path, device="cpu"):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    enc = S4Encoder(c["latent_dim"], c["n_tokens"], c["d_model"],
                    c["d_state"], c["n_layers"], 0.0)
    enc.load_state_dict(blob["encoder"])
    dec = CommonModeDecoder(c["latent_dim"], c["seq_len"])
    dec.load_state_dict(blob["decoder"])
    enc.eval().to(device)
    dec.eval().to(device)
    return enc, dec, blob


@torch.no_grad()
def common_mode_for_group(enc, dec, flux, observed, device="cpu"):
    """flux/observed: (G, L). Returns the predicted common mode, (G, L)."""
    f = torch.as_tensor(flux, dtype=torch.float32, device=device)
    o = torch.as_tensor(observed, dtype=torch.float32, device=device)
    z = enc(f, o)                                  # (G, D)
    g = leave_one_out_mean(z.unsqueeze(0))[0]      # peers only, never the star itself
    return dec(g).cpu().numpy()


def robust_scatter(x):
    """MAD-based scatter, ignoring NaN."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def trend_power(x, k=32):
    """Low-frequency power: variance of the curve after boxcar smoothing.

    The common mode is a slow trend, so this -- not point-to-point MAD -- is the
    quantity subtraction should reduce. MAD is reported alongside to show that
    the high-frequency (per-point) noise is deliberately NOT touched.
    """
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    if good.sum() < k * 2:
        return float("nan")
    y = np.where(good, x, 0.0)
    w = np.convolve(good.astype(float), np.ones(k), "same")
    sm = np.convolve(y, np.ones(k), "same") / np.maximum(w, 1)
    sm = sm[good]
    return float(np.var(sm))


def pick_group(source, group_size=32, area=None, seed=0):
    areas, counts = np.unique(source.area, return_counts=True)
    ok = areas[(counts >= group_size) & (areas >= 0)]
    if area is None:
        area = int(ok[np.argmax(counts[(counts >= group_size) & (areas >= 0)])])
    idx = np.flatnonzero(source.area == area)
    idx = np.random.default_rng(seed).choice(idx, group_size, replace=False)
    return int(area), np.sort(idx)


def make_figure(source, enc, dec, blob, out_path, n_show=4, group_size=32,
                area=None, seed=0, device="cpu"):
    area, idx = pick_group(source, group_size, area, seed)
    flux = source.flux[idx].astype(np.float32)
    obs = source.observed[idx].astype(np.float32)
    tic = source.tic[idx]

    pred = common_mode_for_group(enc, dec, flux, obs, device)
    resid = flux - pred

    m = obs > 0
    before = np.where(m, flux, np.nan)
    model = np.where(m, pred, np.nan)
    after = np.where(m, resid, np.nan)

    # show the stars where the common mode explains the most variance
    tb = np.array([trend_power(before[i]) for i in range(len(before))])
    ta = np.array([trend_power(after[i]) for i in range(len(after))])
    order = np.argsort(-(tb - ta))[:n_show]

    t = np.arange(flux.shape[1])
    fig, axes = plt.subplots(n_show, 2, figsize=(14, 2.5 * n_show), sharex=True)
    fig.suptitle(
        f"Part 1: removing the shared instrumental signal  —  region {area}, "
        f"{group_size} stars, validation split",
        fontsize=13, y=0.995)

    for row, i in enumerate(order):
        a0, a1 = axes[row, 0], axes[row, 1]
        s_before, s_after = robust_scatter(before[i]), robust_scatter(after[i])
        t_before, t_after = trend_power(before[i]), trend_power(after[i])

        a0.plot(t, before[i], lw=0.7, color="#3b4a54", label="observed")
        a0.plot(t, model[i], lw=1.4, color="#c2410c",
                label="common mode from 31 peers")
        a0.set_ylabel(f"TIC {tic[i]}\nnormalised flux", fontsize=8)
        a0.text(0.012, 0.93, f"before   trend power {t_before:.3f}   MAD {s_before:.2f}",
                transform=a0.transAxes, fontsize=8.5, color="#3b4a54", va="top")
        if row == 0:
            a0.legend(loc="lower right", fontsize=8, framealpha=0.9)
            a0.set_title("Before — observed curve with the predicted common mode",
                         fontsize=10)

        a1.axhline(0, lw=0.6, color="#b0b7bd", zorder=0)
        a1.plot(t, after[i], lw=0.7, color="#1d6b52")
        drop = 100 * (1 - t_after / max(t_before, 1e-12))
        a1.text(0.012, 0.93,
                f"after   trend power {t_after:.3f}  ({drop:+.0f}%)   MAD {s_after:.2f}",
                transform=a1.transAxes, fontsize=8.5, color="#1d6b52", va="top")
        if row == 0:
            a1.set_title("After — common mode subtracted", fontsize=10)

        lim = max(np.nanpercentile(np.abs(before[i]), 98) * 1.3, 3.0)
        a0.set_ylim(-lim, lim)
        a1.set_ylim(-lim, lim)
        for a in (a0, a1):
            a.tick_params(labelsize=8)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)

    axes[-1, 0].set_xlabel("cadence (1024-bin shared sector grid)", fontsize=9)
    axes[-1, 1].set_xlabel("cadence (1024-bin shared sector grid)", fontsize=9)

    med_before = float(np.nanmedian(tb))
    med_after = float(np.nanmedian(ta))
    frac_improved = float(np.mean(ta < tb))
    fig.text(0.5, 0.005,
             f"All {group_size} stars in region {area}: median trend power "
             f"{med_before:.3f} -> {med_after:.3f} "
             f"({100*(1-med_after/max(med_before,1e-12)):+.0f}%); "
             f"{100*frac_improved:.0f}% of stars improved.   "
             f"Checkpoint step {blob['step']}, val {blob['val_common']:.3f} "
             f"(short CPU pilot, d_model={blob['cfg']['d_model']}) — "
             f"a pipeline demonstration, not a converged model.",
             ha="center", fontsize=8.5, color="#5b6670")

    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return {"area": area,
            "median_trend_power_before": med_before,
            "median_trend_power_after": med_after,
            "reduction_pct": 100 * (1 - med_after / max(med_before, 1e-12)),
            "frac_stars_improved": frac_improved}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/vicreg_jepa/pilot_real/part1/part1_best.pt")
    ap.add_argument("--npz", default="artifacts/vicreg_jepa/curves.npz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="plots/part1_common_mode.png")
    ap.add_argument("--n-show", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--area", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = RealCurveSource(args.npz, args.split)
    enc, dec, blob = load_part1(args.ckpt)
    stats = make_figure(src, enc, dec, blob, args.out, args.n_show,
                        args.group_size, args.area, args.seed)
    print(stats)
