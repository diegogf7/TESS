"""Checks for the collapse gate and the search's selection rules.

    python -m src.vicreg_jepa.test_search

Pure logic -- no training, no data. Runs in seconds.
"""
import random
import sys

import numpy as np
import torch

from .collapse import CollapseThresholds, describe, reasons, assess, effective_rank
from .config import Part2Config
from .data import random_time_mask
from .search import (SPACE, CONTROL, sample_config, apply_config, rank_trials,
                     expanded_space_suggestion)

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        return fn
    return deco


RNG = np.random.default_rng(0)


def _healthy(n=2048, d=64):
    return RNG.normal(0, 1, (n, d))


def _rank1(n=2048, d=64, noise=0.05):
    return RNG.normal(0, 1, (n, 1)) * np.ones((1, d)) * 0.7 + RNG.normal(0, noise, (n, d))


# ------------------------------------------------------------ collapse gate
@check("the historical failure (std 0.667, rank 1.7) is rejected")
def _():
    s = describe(_rank1())
    assert s["std_median"] > 0.5, "this case must clear the std rule to be meaningful"
    r = reasons(s, CollapseThresholds())
    assert r, "a std-passing, rank-1 latent must still be rejected"
    assert any("effective_rank" in x for x in r)


@check("a healthy latent passes every condition")
def _():
    assert not reasons(describe(_healthy()), CollapseThresholds())


@check("each rejection condition fires on its own failure mode")
def _():
    th = CollapseThresholds()
    cases = {
        "std_median": (RNG.normal(0, 0.05, (2048, 64)), "std_median"),
        "effective_rank": (_rank1(), "effective_rank"),
        "nonfinite": (np.full((512, 64), np.inf), "nonfinite"),
        "dup_pair_frac": (_rank1(noise=0.01), "dup_pair_frac"),
    }
    for label, (z, expect) in cases.items():
        r = reasons(describe(z), th)
        assert any(expect in x for x in r), f"{label}: expected {expect}, got {r}"


@check("effective rank tracks the true dimensionality")
def _():
    for d in (2, 8, 32, 64):
        z = RNG.normal(0, 1, (4096, d)) @ RNG.normal(0, 1, (d, 64))
        er = effective_rank(z)
        assert er <= d + 1, f"rank {er:.1f} exceeds the {d} generating dimensions"


@check("the redundancy gate is scale-free")
def _():
    th = CollapseThresholds()
    base = RNG.normal(0, 1, (4096, 64))
    for scale in (0.8, 1.0, 5.0, 50.0):
        r = reasons(describe(base * scale), th)
        assert not any("offdiag" in x for x in r), \
            f"a healthy latent at scale {scale} tripped the covariance gate: {r}"


@check("the EMA representation is rejected on a single failure")
def _():
    a = assess({"masked_online": _healthy(), "full_online": _healthy(),
                "full_ema": _rank1()}, CollapseThresholds())
    assert a["rejected"] and any(x.startswith("ema:") for x in a["why"])


@check("a transient online failure is tolerated; a persistent one is not")
def _():
    th = CollapseThresholds()
    lanes = {"masked_online": _rank1(), "full_online": _healthy(), "full_ema": _healthy()}
    a1 = assess(lanes, th, online_fail_streak=0)
    assert not a1["rejected"] and a1["online_fail_streak"] == 1
    a3 = assess(lanes, th, online_fail_streak=th.online_persist - 1)
    assert a3["rejected"], "an online failure repeated online_persist times must reject"


@check("the target effective rank is 32 and is reported separately from the gate")
def _():
    th = CollapseThresholds()
    assert th.target_effective_rank == 32.0
    assert th.min_effective_rank == 16.0
    z = RNG.normal(0, 1, (4096, 20)) @ RNG.normal(0, 1, (20, 64))
    z = z / z.std(0, keepdims=True)          # unit-variance dims, rank stays ~17
    a = assess({"masked_online": z, "full_online": z, "full_ema": z}, th)
    assert not a["rejected"], f"rank ~17 should clear the 16 gate: {a['why']}"
    assert not a["meets_target_rank"], "rank ~20 must not count as meeting the 32 target"


# ------------------------------------------------------------ search space
@check("the search space covers every specified range")
def _():
    rng = random.Random(0)
    c = [sample_config(rng, i) for i in range(500)]
    def rng_of(k):
        v = np.array([x[k] for x in c], dtype=float)
        return v[v > 0].min(), v.max()
    assert 0.5 <= rng_of("phi")[0] and rng_of("phi")[1] <= 25.0
    assert 0.1 <= rng_of("lam")[0] and rng_of("lam")[1] <= 50.0
    assert 1.0 <= rng_of("mu")[0] and rng_of("mu")[1] <= 100.0
    assert 0.01 <= rng_of("nu")[0] and rng_of("nu")[1] <= 10.0
    assert 1e-5 <= rng_of("lr")[0] and rng_of("lr")[1] <= 1e-3
    assert 1e-8 <= rng_of("weight_decay")[0] and rng_of("weight_decay")[1] <= 1e-3
    assert any(x["lam"] == 0.0 for x in c), "lambda must include exactly 0"
    assert {x["tau"] for x in c} == {0.99, 0.995, 0.999, 0.9995}
    assert {x["batch_size"] for x in c} == {128, 256, 512}
    assert {x["mask_range"] for x in c} == {(0.2, 0.4), (0.3, 0.5), (0.4, 0.6)}


@check("sampling is biased toward larger covariance weights")
def _():
    rng = random.Random(1)
    c = [sample_config(rng, i) for i in range(500)]
    frac = np.mean([x["nu"] >= 1.0 for x in c])
    assert frac > 0.4, f"only {frac:.2f} of draws had nu >= 1"


@check("the known collapsing control is in the space")
def _():
    assert (CONTROL["phi"], CONTROL["lam"], CONTROL["mu"], CONTROL["nu"]) == \
        (1.0, 25.0, 1.0, 0.01)


@check("a sampled config maps onto Part2Config, mask range included")
def _():
    hp = {"phi": 2.0, "lam": 0.0, "mu": 50.0, "nu": 5.0, "lr": 1e-4,
          "weight_decay": 1e-7, "tau": 0.995, "batch_size": 128,
          "mask_range": (0.4, 0.6), "_name": "t"}
    cfg = apply_config(Part2Config(), hp, steps=123)
    assert (cfg.phi, cfg.lam, cfg.mu, cfg.nu) == (2.0, 0.0, 50.0, 5.0)
    assert (cfg.mask_ratio_min, cfg.mask_ratio_max) == (0.4, 0.6)
    assert cfg.tau == 0.995 and cfg.batch_size == 128 and cfg.steps == 123


# ------------------------------------------------------------ selection
def _t(name, eligible=True, acc=0.7, corr=0.2, leak=0.5, er=40.0, inv=0.01):
    return {"name": name, "eligible": eligible, "metrics": {
        "physics_bacc": acc, "mean_abs_corr_with_systematics": corr,
        "camera_bacc": leak, "ccd_bacc": leak, "area_bacc": leak,
        "effective_rank": er, "offdiag_cov_rms": 0.1, "val_inv": inv}}


@check("no eligible trial -> refuses to select anything")
def _():
    o, n = rank_trials([_t("a", eligible=False), _t("b", eligible=False)])
    assert o == [], "a collapsed trial must never be selected"
    assert "no trial produced" in n["reason"]


@check("the least-collapsed trial is never promoted")
def _():
    # 'near' has by far the best metrics but failed the gate; it must not win.
    o, _ = rank_trials([_t("near", eligible=False, acc=0.99, corr=0.0, er=63.0)])
    assert o == [], "an ineligible trial won on metrics -- selection is unsafe"


@check("accuracy leads, correlation breaks ties inside the 0.01 band")
def _():
    o, n = rank_trials([_t("hi_acc_hi_corr", acc=0.700, corr=0.40),
                        _t("band_lo_corr", acc=0.695, corr=0.05),
                        _t("band_mid_corr", acc=0.693, corr=0.20),
                        _t("low_acc_lo_corr", acc=0.600, corr=0.01)])
    assert n["n_in_accuracy_band"] == 3
    assert o[0]["name"] == "band_lo_corr"
    assert o[-1]["name"] == "low_acc_lo_corr", "low accuracy must not win on correlation"


@check("leakage, then effective rank, then invariance break deeper ties")
def _():
    o, _ = rank_trials([_t("leaky", acc=0.70, corr=0.10, leak=0.9, er=40),
                        _t("clean", acc=0.70, corr=0.10, leak=0.2, er=40)])
    assert o[0]["name"] == "clean"
    o, _ = rank_trials([_t("lowrank", acc=0.70, corr=0.10, leak=0.5, er=20),
                        _t("highrank", acc=0.70, corr=0.10, leak=0.5, er=55)])
    assert o[0]["name"] == "highrank"
    o, _ = rank_trials([_t("hi_inv", acc=0.70, corr=0.1, leak=0.5, er=40, inv=0.9),
                        _t("lo_inv", acc=0.70, corr=0.1, leak=0.5, er=40, inv=0.001)])
    assert o[0]["name"] == "lo_inv"


@check("missing PhyTS labels are reported, not silently ignored")
def _():
    o, n = rank_trials([_t("x", acc=float("nan")), _t("y", acc=float("nan"), corr=0.05)])
    assert "warning" in n and "criterion 1" in n["warning"]
    assert o[0]["name"] == "y"


@check("failure reports an expanded space and never a relaxed threshold")
def _():
    s = expanded_space_suggestion()
    assert "NOT lowered" in s["note"]
    assert {"mu", "nu", "lam", "lr", "tau"} <= set(s["widen"])


# ------------------------------------------------------------ determinism
@check("fixed masking seeds give identical masks across trials")
def _():
    f, o = torch.zeros(64, 1024), torch.ones(64, 1024)
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    assert torch.equal(random_time_mask(f, o, generator=g1),
                       random_time_mask(f, o, generator=g2))
    assert not torch.equal(random_time_mask(f, o), random_time_mask(f, o))


def run():
    print()
    for name in ():
        pass
    print(f"{len(PASS) + len(FAIL)} checks run")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED:")
        for n, e in FAIL:
            print(f"  - {n}: {e}")
        return 1
    print(f"ALL {len(PASS)} SEARCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run())
