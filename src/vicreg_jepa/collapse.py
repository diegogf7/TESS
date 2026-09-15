"""Collapse detection: the gate every checkpoint must pass.

Standard deviation alone is never proof that collapse was avoided. A real pilot
produced a latent with median std 0.667 -- clearing a std-only rule -- whose
effective rank was 1.7 of 64: the variance hinge had been satisfied by copying
essentially the same signal into every dimension. Every threshold below exists
because std missed something.

Three representations are measured separately at each validation:
  masked_online  encoder(masked view)      -- what the VICReg terms act on
  full_online    encoder(full curve)       -- diagnostic
  full_ema       ema_encoder(full curve)   -- THE PRIMARY REPRESENTATION

A checkpoint is rejected when the primary representation fails any condition, or
when the online representation fails on several consecutive validations.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class CollapseThresholds:
    min_effective_rank: float = 16.0      # of 64, per the search specification
    min_std_median: float = 0.5
    std_floor: float = 0.5
    max_frac_std_below_floor: float = 0.20
    # Redundancy is judged scale-free: off-diagonal covariance relative to the
    # mean diagonal. An absolute cap is kept purely as an explosion guard, so a
    # latent that is merely larger in scale is not mistaken for a collapsed one.
    max_offdiag_cov_ratio: float = 0.5
    max_offdiag_cov_rms: float = 100.0    # "exploding off-diagonal covariance"
    dup_corr: float = 0.95                # "nearly identical latent dimensions"
    max_dup_pair_frac: float = 0.05
    online_persist: int = 3               # consecutive online failures = reject
    target_effective_rank: float = 32.0   # final model target, never lowered

    def to_dict(self):
        return asdict(self)


def effective_rank(z):
    """exp(entropy of the normalised covariance spectrum). D means no collapse."""
    z = np.asarray(z, dtype=np.float64)
    c = np.cov(z - z.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    total = ev.sum()
    if not np.isfinite(total) or total <= 0:
        return 0.0
    p = ev / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def duplicate_pair_fraction(z, dup_corr=0.95):
    """Fraction of distinct dimension pairs whose |Pearson r| exceeds dup_corr."""
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(0)
    sd = zc.std(0)
    live = sd > 1e-12
    if live.sum() < 2:
        return 1.0
    corr = np.corrcoef(zc[:, live], rowvar=False)
    d = corr.shape[0]
    iu = np.triu_indices(d, k=1)
    vals = np.abs(corr[iu])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    # dimensions that are dead (zero variance) count as duplicates of each other
    dead = int((~live).sum())
    total_pairs = z.shape[1] * (z.shape[1] - 1) / 2
    dup = float((vals > dup_corr).sum()) + dead * (dead - 1) / 2 + dead * int(live.sum())
    return float(dup / max(total_pairs, 1))


def describe(z):
    """Every collapse statistic for one representation."""
    z = np.asarray(z, dtype=np.float64)
    finite = bool(np.isfinite(z).all())
    if not finite:
        return {"dims": int(z.shape[1]), "finite": False, "std_min": float("nan"),
                "std_median": float("nan"), "std_max": float("nan"),
                "frac_std_below_floor": 1.0, "offdiag_cov_rms": float("inf"),
                "offdiag_cov_absmax": float("inf"), "diag_cov_mean": float("nan"),
                "offdiag_cov_ratio": float("inf"),
                "effective_rank": 0.0, "dup_pair_frac": 1.0}
    std = z.std(axis=0)
    c = np.cov(z - z.mean(0), rowvar=False)
    off = c - np.diag(np.diag(c))
    diag_mean = float(np.diag(c).mean())
    off_rms = float(np.sqrt((off ** 2).mean()))
    return {
        "dims": int(z.shape[1]),
        "finite": True,
        "std_min": float(std.min()),
        "std_median": float(np.median(std)),
        "std_max": float(std.max()),
        "frac_std_below_floor": float((std < 0.5).mean()),
        "offdiag_cov_rms": off_rms,
        "offdiag_cov_absmax": float(np.abs(off).max()),
        "diag_cov_mean": diag_mean,
        "offdiag_cov_ratio": float(off_rms / max(diag_mean, 1e-12)),
        "effective_rank": effective_rank(z),
        "dup_pair_frac": duplicate_pair_fraction(z),
    }


def reasons(stats, th: CollapseThresholds):
    """Which rejection conditions this representation trips."""
    out = []
    if not stats["finite"]:
        out.append("nonfinite")
        return out
    if stats["effective_rank"] < th.min_effective_rank:
        out.append(f"effective_rank {stats['effective_rank']:.2f}<{th.min_effective_rank:g}")
    if stats["std_median"] < th.min_std_median:
        out.append(f"std_median {stats['std_median']:.3f}<{th.min_std_median:g}")
    if stats["frac_std_below_floor"] > th.max_frac_std_below_floor:
        out.append(f"frac_std_below_{th.std_floor:g} "
                   f"{stats['frac_std_below_floor']:.2f}>{th.max_frac_std_below_floor:g}")
    if stats["offdiag_cov_ratio"] > th.max_offdiag_cov_ratio:
        out.append(f"offdiag_cov_ratio {stats['offdiag_cov_ratio']:.3f}"
                   f">{th.max_offdiag_cov_ratio:g}")
    if stats["offdiag_cov_rms"] > th.max_offdiag_cov_rms:
        out.append(f"offdiag_cov_rms {stats['offdiag_cov_rms']:.3g}"
                   f">{th.max_offdiag_cov_rms:g} (exploding)")
    if stats["dup_pair_frac"] > th.max_dup_pair_frac:
        out.append(f"dup_pair_frac {stats['dup_pair_frac']:.3f}>{th.max_dup_pair_frac:g}")
    return out


def assess(latents, th: CollapseThresholds, online_fail_streak=0):
    """Judge one validation checkpoint across all three representations.

    latents: {"masked_online": arr, "full_online": arr, "full_ema": arr}

    The primary representation (full_ema) is rejected on any single failure.
    The online representations are rejected only when they fail on
    `th.online_persist` consecutive validations -- a transient online wobble
    while the EMA target stays healthy is not yet collapse.
    """
    stats = {k: describe(v) for k, v in latents.items()}
    rsn = {k: reasons(s, th) for k, s in stats.items()}

    primary_bad = bool(rsn["full_ema"])
    online_bad = bool(rsn["masked_online"]) or bool(rsn["full_online"])
    streak = online_fail_streak + 1 if online_bad else 0
    persistent_online = streak >= th.online_persist

    rejected = primary_bad or persistent_online
    why = []
    if primary_bad:
        why += [f"ema:{r}" for r in rsn["full_ema"]]
    if persistent_online:
        why += [f"online x{streak}:{r}" for r in (rsn["masked_online"] or rsn["full_online"])]

    return {
        "stats": stats,
        "reasons": rsn,
        "rejected": rejected,
        "why": why,
        "online_fail_streak": streak,
        "primary_effective_rank": stats["full_ema"]["effective_rank"],
        "meets_target_rank": (stats["full_ema"]["effective_rank"]
                              >= th.target_effective_rank),
    }
