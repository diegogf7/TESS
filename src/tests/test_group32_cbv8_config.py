# All this code is from Claude
"""Contracts for the decoupled group-CBV config: GROUP_SIZE=32, CBV_RANK=8,
MIN_VALID_STARS=16. Proves group size, basis rank, and cadence validity are
three INDEPENDENT knobs. Synthetic only; no cluster, checkpoints, or test TICs.

Run: python -m src.tests.test_group32_cbv8_config
"""

import glob
import json
import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    group_statistics,
)
from src.instrument_v2.regional_cbv import (
    build_or_load_area_bases,
    train_tic_hash,
    uncentered_area_basis,
)
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.tests.test_area_commonmode_jepa import frame_with_areas

GROUP_SIZE = 32
CBV_RANK = 8
MIN_VALID_STARS = 16


def _cfg_dataset(n_per_chip=64):
    """One area per chip (ring 1) with exactly `n_per_chip` stars -> supports
    a 2 * GROUP_SIZE = 64 draw. Built with the explicit min_valid=16."""
    frame = frame_with_areas(n_per_chip=n_per_chip, n_rings=1)
    t0 = min(float(np.min(t)) for t in frame["time"])
    t1 = max(float(np.max(t)) for t in frame["time"])
    return Sector14GroupStatDataset(frame, set(frame["TIC"].astype(str)),
                                    (t0, t1), "area", GROUP_SIZE,
                                    grid_length=64, min_valid=MIN_VALID_STARS)


def test_stage_a_samples_64_unique_two_disjoint_32():
    ds = _cfg_dataset()
    np.random.seed(0)
    got = ds.sample_disjoint_same_group()          # needs >= 2 * GROUP_SIZE stars
    assert got is not None
    rows_a, rows_b, _area = got
    assert len(rows_a) == GROUP_SIZE and len(rows_b) == GROUP_SIZE
    assert not (set(rows_a.tolist()) & set(rows_b.tolist()))          # disjoint
    assert len(set(rows_a.tolist()) | set(rows_b.tolist())) == 2 * GROUP_SIZE


def test_stage_b_samples_33_unique_one_context_plus_32_targets():
    ds = _cfg_dataset()
    np.random.seed(1)
    context, targets, _group = ds._sample_item()
    assert len(targets) == GROUP_SIZE                                # 32 targets
    assert context not in set(targets.tolist())                      # context excluded
    assert len({context} | set(targets.tolist())) == GROUP_SIZE + 1  # 33 unique


def test_cadence_invalid_at_15_valid_at_16():
    L = 3
    flux = np.ones((GROUP_SIZE, L), dtype=np.float32)
    mask = np.zeros((GROUP_SIZE, L), dtype=np.float32)
    mask[:15, 0] = 1.0        # 15 observed -> below MIN_VALID_STARS -> invalid
    mask[:16, 1] = 1.0        # 16 observed -> exactly MIN_VALID_STARS -> valid
    mask[:, 2] = 1.0          # all 32 observed -> valid
    _, _, valid, n_obs = group_statistics(flux, mask, MIN_VALID_STARS)
    assert n_obs[0] == 15 and valid[0] == 0.0
    assert n_obs[1] == 16 and valid[1] == 1.0
    assert valid[2] == 1.0


def test_cbv_basis_shape_is_1024_by_8():
    rng = np.random.default_rng(0)
    medians = rng.normal(size=(CBV_RANK, 1024)).astype(np.float32)   # >= 8 groups
    valids = np.ones((CBV_RANK, 1024), dtype=np.float32)
    B = uncentered_area_basis(medians, valids, CBV_RANK)
    assert B.shape == (1024, CBV_RANK)                               # rank independent of group size


def test_weight_decoder_dim_8_direct_dim_1024():
    assert build_decoder(CBV_RANK)(torch.randn(2, 16, 16)).shape == (2, CBV_RANK)
    assert build_decoder()(torch.randn(2, 16, 16)).shape == (2, 1024)


def test_val_test_tics_never_enter_cbv_basis():
    rng = np.random.default_rng(0)
    n_train, L = 8 * GROUP_SIZE, 64          # 256 train stars -> 8 groups -> rank 8
    X = rng.normal(size=(n_train, L)).astype(np.float32)
    M = np.ones((n_train, L), dtype=np.float32)
    areas = np.full(n_train, 111, dtype=np.int64)
    train = [f"TRN{i}" for i in range(n_train)]
    held = [f"HELD{i}" for i in range(GROUP_SIZE)]      # val/test TICs, never passed in
    tmp = tempfile.mkdtemp()
    build_or_load_area_bases(X, M, areas, train, CBV_RANK, tmp, GROUP_SIZE,
                             MIN_VALID_STARS)
    meta = json.load(open(glob.glob(os.path.join(tmp, "*.json"))[0]))
    assert meta["train_tic_hash"] == train_tic_hash(train)
    assert meta["train_tic_hash"] != train_tic_hash(sorted(train + held))
    assert meta["k"] == CBV_RANK and meta["group_size"] == GROUP_SIZE
    assert meta["min_valid"] == MIN_VALID_STARS


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} GROUP32-CBV8 CONFIG TESTS PASSED")
