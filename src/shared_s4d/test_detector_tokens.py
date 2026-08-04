from __future__ import annotations
"""Tests for detector-nearest grouping + time-resolved temporal tokens.
    python -m src.shared_s4d.test_detector_tokens
"""
import numpy as np
import torch

from src.shared_s4d.dataset import AreaGroupLOODataset
from src.shared_s4d.model import build_model
from src.models.s4d import masked_token_pool
from src.shared_s4d.correction_losses import windowed_group_cov_loss, soft_cap_size


def _synth_area_dataset(gs=16, per_area=30, L=64, seed=0):
    """Two areas with distinct camera/ccd (area = camera*100 + ccd*10 + bin) and random
    detector STAR_X/STAR_Y. X/M are dummies -- grouping never reads them."""
    rng = np.random.default_rng(seed)
    X, M, areas, tics, detxy = [], [], [], [], []
    t = 0
    for a in (110, 120):                                     # cam1/ccd1, cam1/ccd2
        for _ in range(per_area):
            X.append(np.zeros(L, np.float32)); M.append(np.ones(L, np.float32))
            areas.append(a); tics.append(str(t)); t += 1
            detxy.append(rng.uniform(0, 2048, size=2))
    return (np.asarray(X), np.asarray(M), np.asarray(areas), np.asarray(tics),
            np.asarray(detxy, np.float64))


# ---- grouping ---------------------------------------------------------------
def test_detector_neighbors_are_the_15_closest():
    X, M, areas, tics, detxy = _synth_area_dataset()
    ds = AreaGroupLOODataset(X, M, areas, tics, n_stars=1000, group_size=16,
                             require_full=False, grouping_mode="detector_nearest", detxy=detxy)
    assert 0 < len(ds.items) <= 60                            # <=30 anchors x 2 areas (exact dups removed)
    for rows, a in ds.items:
        anchor = rows[0]                                      # anchor is its own nearest (dist 0)
        pool = np.where(areas == a)[0]
        d = np.sqrt(((detxy[pool] - detxy[anchor]) ** 2).sum(1))
        want = set(pool[np.argsort(d, kind="stable")[:16]].tolist())
        assert set(rows.tolist()) == want                    # exactly anchor + 15 nearest by Euclidean
        assert len(rows) == 16 == len(set(rows.tolist()))    # 16 unique


def test_no_group_crosses_area_camera_ccd_or_split():
    X, M, areas, tics, detxy = _synth_area_dataset()
    ds = AreaGroupLOODataset(X, M, areas, tics, n_stars=1000, group_size=16,
                             require_full=False, grouping_mode="detector_nearest", detxy=detxy)
    ds.assert_contracts()
    pool_by_area = {a: set(np.where(areas == a)[0].tolist()) for a in np.unique(areas)}
    for rows, a in ds.items:
        ga = set(areas[rows].tolist())
        assert ga == {a}                                     # single area
        assert len({x // 100 for x in ga}) == 1              # single camera (area//100)
        assert len({(x // 10) % 10 for x in ga}) == 1        # single ccd ((area//10)%10)
        assert set(rows.tolist()) <= pool_by_area[a]         # never leaves this split's star pool


def test_detector_nearest_requires_detxy():
    X, M, areas, tics, _ = _synth_area_dataset()
    try:
        AreaGroupLOODataset(X, M, areas, tics, group_size=16, grouping_mode="detector_nearest")
        raise AssertionError("expected RuntimeError when DETECTOR_X/DETECTOR_Y missing")
    except RuntimeError as e:
        assert "DETECTOR_X" in str(e) or "Regenerate" in str(e)


def test_real_detector_group_from_parquet():
    """Load the merged xy parquet (cluster only) and build one REAL detector-nearest group.
    Skips where the parquet is absent (e.g. local runs)."""
    import os
    import pandas as pd
    path = os.environ.get("DETXY_PARQUET",
                          "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2_xy.parquet")
    if not os.path.exists(path):
        print("  (skip test_real_detector_group_from_parquet -- no xy parquet here)"); return
    df = pd.read_parquet(path, columns=["TIC", "camera", "ccd", "DETECTOR_X", "DETECTOR_Y"])
    df = df.dropna(subset=["DETECTOR_X", "DETECTOR_Y"]).drop_duplicates("TIC")
    (cam, ccd) = df.groupby(["camera", "ccd"]).size().sort_values().index[-1]     # densest chip
    sub = df[(df.camera == cam) & (df.ccd == ccd)].head(300).reset_index(drop=True)
    n = len(sub)
    detxy = sub[["DETECTOR_X", "DETECTOR_Y"]].to_numpy(float)
    areas = np.full(n, int(cam) * 100 + int(ccd) * 10, np.int64)                  # single pool
    ds = AreaGroupLOODataset(np.zeros((n, 8), np.float32), np.ones((n, 8), np.float32),
                             areas, sub["TIC"].to_numpy().astype(str), n_stars=1000, group_size=16,
                             require_full=False, grouping_mode="detector_nearest", detxy=detxy)
    assert len(ds.items) > 0
    rows, _a = ds.items[0]; anchor = rows[0]
    d = np.sqrt(((detxy - detxy[anchor]) ** 2).sum(1))
    assert set(rows.tolist()) == set(np.argsort(d, kind="stable")[:16].tolist()) and len(rows) == 16
    print(f"  real detector group OK (cam{cam} ccd{ccd}, {n} stars)")


# ---- temporal-token pooling -------------------------------------------------
def test_masked_pool_ignores_missing_cadences():
    B, L, D, N = 2, 1024, 4, 8; blk = L // N
    x = torch.randn(B, L, D); mask = torch.ones(B, L)
    mask[:, blk // 2:blk] = 0                                 # mask second half of block 0
    pooled = masked_token_pool(x, mask, N)                    # (B, N, D)
    manual0 = x[:, :blk // 2, :].mean(1)                      # mean over the VALID cadences only
    assert torch.allclose(pooled[:, 0, :], manual0, atol=1e-5)
    x2 = x.clone(); x2[:, blk // 2:blk, :] = 999.0            # change MASKED values -> token unchanged
    assert torch.allclose(masked_token_pool(x2, mask, N)[:, 0, :], pooled[:, 0, :], atol=1e-5)


def test_empty_block_returns_zero_and_no_nan():
    B, L, D, N = 2, 1024, 4, 8; blk = L // N
    x = torch.randn(B, L, D); mask = torch.ones(B, L); mask[:, :blk] = 0   # block 0 fully missing
    pooled = masked_token_pool(x, mask, N)
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled[:, 0, :], torch.zeros(B, D))              # zero token
    model = build_model(n_tokens=8, token_dim=32).eval()                   # full model stays finite
    c, z = model(torch.randn(B, L), mask)
    assert torch.isfinite(c).all() and torch.isfinite(z).all()


# ---- model shapes / gradients / inference -----------------------------------
def test_shapes_for_1_and_8_tokens():
    x = torch.randn(4, 1024); m = torch.ones(4, 1024)
    for nt, zdim in [(1, 32), (8, 256)]:
        model = build_model(n_tokens=nt, token_dim=32).eval()
        c, z = model(x, m)
        assert c.shape == (4, 1024) and z.shape == (4, zdim), (nt, c.shape, z.shape)
        assert torch.isfinite(c).all() and torch.isfinite(z).all()


def test_single_token_baseline_unchanged():
    model = build_model(n_tokens=1, token_dim=32).eval()
    c, z = model(torch.randn(2, 1024), torch.ones(2, 1024))
    assert z.shape == (2, 32) and c.shape == (2, 1024) and model.latent_dim == 32


def test_gradients_reach_encoder_projection_decoder():
    model = build_model(n_tokens=8, token_dim=32).train()
    c, _ = model(torch.randn(3, 1024), torch.ones(3, 1024))
    c.sum().backward()
    assert model.encoder.encoder.weight.grad is not None       # S4D input Linear(1->d_model)
    assert model.encoder.decoder.weight.grad is not None       # shared token projection Linear(d_model->token_dim)
    assert model.decoder[0].weight.grad is not None            # correction MLP
    assert model.encoder.s4_layers[0].kernel.log_dt.grad is not None   # an S4D kernel param


def test_single_curve_inference():
    model = build_model(n_tokens=8, token_dim=32).eval()
    with torch.no_grad():
        c, z = model(torch.randn(1, 1024), torch.ones(1, 1024))
    assert c.shape == (1, 1024) and z.shape == (1, 256) and torch.isfinite(c).all()


def test_one_backward_one_step_per_group():
    model = build_model(n_tokens=8, token_dim=32)
    steps = {"n": 0}
    class CountAdamW(torch.optim.AdamW):
        def step(self, *a, **k):
            steps["n"] += 1; return super().step(*a, **k)
    opt = CountAdamW(model.parameters(), lr=1e-3)
    x = torch.randn(16, 1024); m = torch.ones(16, 1024)
    for _ in range(3):
        opt.zero_grad()
        c, _ = model(x, m); r = x - c
        (windowed_group_cov_loss(r, x, m) + 0.1 * soft_cap_size(c, x, m)).backward()
        opt.step()
    assert steps["n"] == 3


def run():
    torch.manual_seed(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("ALL DETECTOR-NEAREST + TEMPORAL-TOKEN TESTS PASSED")


if __name__ == "__main__":
    run()
