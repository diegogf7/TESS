# All this code is from Claude
"""Unit tests for the Sector-14 chip-pair JEPA data pipeline.

Synthetic data only. Run: python -m src.tests.test_sector14_jepa
"""

import os
import tempfile

import numpy as np
import pandas as pd

from src.instrument_v2.sector14_dataset import (
    Sector14ChipPairDataset,
    carve_validation,
    ensure_splits,
    grid_frame,
    shared_grid_bin,
)


def synthetic_frame(n_per_chip=6, n_cad=120, with_flags=True, garbage_cal=False):
    rng = np.random.default_rng(0)
    rows, tic = [], 0
    time_all = 1683.0 + np.arange(n_cad) * 0.0208
    for camera in range(1, 5):
        for ccd in range(1, 5):
            for _ in range(n_per_chip):
                keep = rng.random(n_cad) > 0.15
                row = {"TIC": f"TIC{tic}", "sector": 14, "camera": camera, "ccd": ccd,
                       "time": time_all[keep],
                       "flux": rng.normal(1000, 30, int(keep.sum()))}
                if with_flags:
                    row["TESS_flags"] = (rng.random(int(keep.sum())) < 0.3).astype(int)
                    row["TGLC_flags"] = (rng.random(int(keep.sum())) < 0.2).astype(int)
                if garbage_cal:
                    row["flux_cal"] = np.full(int(keep.sum()), 1e12)
                rows.append(row)
                tic += 1
    return pd.DataFrame(rows)


def _dataset(df, arm="shared", tics=None):
    tics = tics if tics is not None else set(df["TIC"].astype(str))
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    return Sector14ChipPairDataset(df, tics, arm, (t0, t1))


def test_pairs_are_different_tics_same_chip():
    ds = _dataset(synthetic_frame())
    np.random.seed(0)
    for _ in range(500):
        a, b, chip = ds._sample_pair()
        assert ds.tics[a] != ds.tics[b], "pair reuses the same TIC"
        assert ds.chips[a] == ds.chips[b] == chip, "pair crosses chips"


def test_chip_sampling_is_balanced():
    """Uniform-over-chips sampling despite a 20:2 star imbalance."""
    df = synthetic_frame(n_per_chip=2)
    heavy = synthetic_frame(n_per_chip=18)
    heavy = heavy[(heavy["camera"] == 1)]                    # camera 1 dominates rows
    heavy["TIC"] = "H" + heavy["TIC"].astype(str)
    df = pd.concat([df, heavy], ignore_index=True)
    ds = _dataset(df)
    np.random.seed(1)
    n_draws = 8000
    counts = np.zeros(16)
    for _ in range(n_draws):
        _, _, chip = ds._sample_pair()
        counts[chip] += 1
    expected = n_draws / 16
    assert counts.min() > 0.8 * expected and counts.max() < 1.2 * expected, \
        f"chip sampling unbalanced: {counts.tolist()}"


def test_train_val_test_disjoint():
    tics = [f"TIC{i}" for i in range(200)]
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "split_train_tics.txt"), "w") as fh:
            fh.write("\n".join(tics[:160]))
        with open(os.path.join(tmp, "split_test_tics.txt"), "w") as fh:
            fh.write("\n".join(tics[160:]))
        art = os.path.join(tmp, "exp")
        train, val, test = ensure_splits(tmp, art)
        assert not (train & val) and not (train & test) and not (val & test)
        assert len(val) == 16 and len(train) == 144 and len(test) == 40
        train2, val2, test2 = ensure_splits(tmp, art)      # reload -> identical
        assert train == train2 and val == val2 and test == test2
    # carving is deterministic under seed 43
    a = carve_validation({f"T{i}" for i in range(100)})
    b = carve_validation({f"T{i}" for i in range(100)})
    assert a == b


def test_identical_times_map_to_identical_shared_bins():
    t0, t1 = 1683.0, 1710.0
    times = np.array([1683.0, 1690.1234, 1700.5678, 1709.9999])
    b1 = shared_grid_bin(times, t0, t1)
    b2 = shared_grid_bin(times.copy(), t0, t1)
    assert np.array_equal(b1, b2)
    # the same absolute time inside two different stars' arrays -> same bin
    star_a = np.array([1690.1234, 1695.0])
    star_b = np.array([1685.0, 1690.1234])
    assert shared_grid_bin(star_a, t0, t1)[0] == shared_grid_bin(star_b, t0, t1)[1]


def test_only_raw_flux_column_used():
    """Garbage (or absent) flux_cal must not change any model input."""
    df_plain = synthetic_frame(garbage_cal=False)
    df_garbage = synthetic_frame(garbage_cal=True)          # same rng -> same flux
    ds1 = _dataset(df_plain)
    ds2 = _dataset(df_garbage)
    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.M, ds2.M)
    df_nocal = df_plain.drop(columns=[c for c in df_plain.columns if "cal" in c],
                             errors="ignore")
    ds3 = _dataset(df_nocal)                                # flux_cal not required
    np.testing.assert_array_equal(ds1.X, ds3.X)


def test_quality_flags_do_not_remove_observations():
    """Primary arm keeps every finite raw-flux cadence, flagged or not."""
    df = synthetic_frame(with_flags=True)
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    X, M = grid_frame(df, "legacy", (t0, t1))
    for i in range(len(df)):
        n_finite = int(np.isfinite(np.asarray(df["flux"].iloc[i], dtype=float)).sum())
        # legacy grid marks bins within 3 cadences of a sample as observed;
        # with >=85% coverage the observed count must reflect ALL cadences,
        # not just quality-clean ones (~56% here). A flag-filtered mask would
        # be far sparser.
        assert M[i].sum() >= 0.8 * min(n_finite, X.shape[1] * 0.9), \
            f"row {i}: mask too sparse -- flags may have removed observations"
    # direct check on the shared grid: observed bins == bins hit by ALL cadences
    Xs, Ms = grid_frame(df.iloc[:3], "shared", (t0, t1))
    for i in range(3):
        bins = shared_grid_bin(np.asarray(df["time"].iloc[i], dtype=float), t0, t1)
        assert Ms[i].sum() == len(np.unique(bins)), "shared mask dropped flagged cadences"


def test_test_tics_never_enter_pretraining():
    df = synthetic_frame()
    all_tics = sorted(set(df["TIC"].astype(str)))
    train_tics = set(all_tics[:70])
    test_tics = set(all_tics[70:])
    ds = _dataset(df, tics=train_tics)
    assert set(ds.tics) <= train_tics, "dataset contains non-train TICs"
    assert not (set(ds.tics) & test_tics), "test TIC leaked into pretraining dataset"
    np.random.seed(2)
    for _ in range(300):
        a, b, _ = ds._sample_pair()
        assert ds.tics[a] in train_tics and ds.tics[b] in train_tics


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
