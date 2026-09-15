"""Collapse diagnostics, disentanglement metrics and frozen-probe evaluation.

Every probe here fits on TRAIN only (scaler included) and reports balanced
accuracy elsewhere, so numbers across representations stay comparable.
"""
import json
import numpy as np
import torch

from .data import random_time_mask
from .losses import cross_correlation_loss


# ------------------------------------------------------------ collapse
def effective_rank(z):
    """exp(entropy of the normalised covariance spectrum). D means no collapse."""
    z = np.asarray(z, dtype=np.float64)
    c = np.cov(z - z.mean(0), rowvar=False)
    ev = np.linalg.eigvalsh(c)
    ev = np.clip(ev, 0, None)
    total = ev.sum()
    if total <= 0:
        return 0.0
    p = ev / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_report(z):
    z = np.asarray(z, dtype=np.float64)
    std = z.std(axis=0)
    c = np.cov(z - z.mean(0), rowvar=False)
    off = c - np.diag(np.diag(c))
    return {
        "dims": int(z.shape[1]),
        "std_min": float(std.min()),
        "std_median": float(np.median(std)),
        "std_max": float(std.max()),
        "frac_std_below_0.5": float((std < 0.5).mean()),
        "offdiag_cov_rms": float(np.sqrt((off ** 2).mean())),
        "offdiag_cov_absmax": float(np.abs(off).max()),
        "effective_rank": effective_rank(z),
    }


# Effective rank below this fraction of the latent width counts as collapse.
MIN_ERANK_FRAC = 0.25


def collapse_reasons(report, min_erank_frac=MIN_ERANK_FRAC):
    """Which rejection conditions fired, if any.

    The first two are the spec's rule. The third was added after a real-data
    pilot produced a latent with median std 0.667 -- clearing the spec's rule --
    whose effective rank was 1.7 of 64 and whose off-diagonal covariance had
    exploded to 11.7. The variance hinge can satisfy a per-dimension std floor
    by inflating dimensions that all point the same way, so a std-only rule is
    necessary but not sufficient. Effective rank catches that; std cannot.
    """
    reasons = []
    if report["std_median"] < 0.5:
        reasons.append("std_median<0.5")
    if report["frac_std_below_0.5"] > 0.20:
        reasons.append("frac_std_below_0.5>0.20")
    floor = min_erank_frac * report["dims"]
    if report["effective_rank"] < floor:
        reasons.append(f"effective_rank<{floor:.1f}")
    return reasons


def is_collapsed(report, min_erank_frac=MIN_ERANK_FRAC):
    """Rejection rule: the spec's two std conditions plus an effective-rank floor."""
    return bool(collapse_reasons(report, min_erank_frac))


# ------------------------------------------------------------ probes
def probe(z_train, y_train, z_eval, y_eval, seed=0, max_iter=2000):
    """Frozen-representation linear probe -> balanced accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = np.asarray(y_train)
    y_eval = np.asarray(y_eval)
    keep_tr = y_train >= 0
    keep_ev = y_eval >= 0
    if keep_tr.sum() < 10 or keep_ev.sum() < 10 or len(np.unique(y_train[keep_tr])) < 2:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, multi_class="auto", random_state=seed))
    clf.fit(z_train[keep_tr], y_train[keep_tr])
    return float(balanced_accuracy_score(y_eval[keep_ev], clf.predict(z_eval[keep_ev])))


def mean_abs_corr(a, b):
    """Mean |Pearson r| over the a x b cross-correlation matrix."""
    return float(cross_correlation_loss(torch.as_tensor(np.asarray(a), dtype=torch.float32),
                                        torch.as_tensor(np.asarray(b), dtype=torch.float32)))


# ------------------------------------------------------------ encoding
@torch.no_grad()
def encode_source(encoder, source, device, batch_size=256, mask_view=False,
                  cfg=None, seed=0):
    """Run an encoder over every curve in a source. Returns (N, D) float64.

    mask_view=False is the evaluation default: the FULL observed curve, which
    is what the EMA encoder sees during training.
    """
    encoder.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    flux = torch.from_numpy(np.asarray(source.flux, dtype=np.float32))
    obs = torch.from_numpy(np.asarray(source.observed, dtype=np.float32))
    out = []
    for i in range(0, len(flux), batch_size):
        f = flux[i:i + batch_size].to(device)
        o = obs[i:i + batch_size].to(device)
        if mask_view:
            torch.manual_seed(int(gen.initial_seed()) + i)
            v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        else:
            v = o
        out.append(encoder(f, v, pool_mask=o).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ------------------------------------------------------------ full report
def evaluate_representation(name, encoder, part1_encoder, sources, device,
                            cfg=None, seed=0, batch_size=256):
    """sources: {"train": RealCurveSource, "val": ..., "test": ...} (test optional).

    Returns every metric the spec asks for, per evaluation split.
    """
    z = {k: encode_source(encoder, s, device, batch_size) for k, s in sources.items()}
    zs = {k: encode_source(part1_encoder, s, device, batch_size) for k, s in sources.items()}

    rep = {"representation": name, "seed": seed, "splits": {}}
    for k, src in sources.items():
        if k == "train":
            continue
        r = collapse_report(z[k])
        r["collapsed"] = is_collapsed(r)
        r["collapse_reasons"] = collapse_reasons(r)
        r["mean_abs_corr_with_systematics"] = mean_abs_corr(z[k], zs[k])
        r["physics_bacc"] = probe(z["train"], sources["train"].label,
                                  z[k], src.label, seed)
        r["physics_bacc_from_systematics"] = probe(
            zs["train"], sources["train"].label, zs[k], src.label, seed)
        for field in ("camera", "ccd", "sector", "area"):
            r[f"{field}_bacc"] = probe(z["train"], getattr(sources["train"], field),
                                       z[k], getattr(src, field), seed)
        rep["splits"][k] = r
    return rep


def write_report(report, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return path


def summarise_seeds(reports, split="val", keys=("physics_bacc",
                                                "mean_abs_corr_with_systematics",
                                                "effective_rank", "std_median")):
    """mean +/- sd across seeds for one representation."""
    out = {}
    for k in keys:
        vals = [r["splits"][split][k] for r in reports
                if split in r["splits"] and k in r["splits"][split]]
        vals = [v for v in vals if v == v]      # drop NaN
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
                      "n_seeds": len(vals)}
    return out
