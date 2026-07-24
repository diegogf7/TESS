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
                if with_flags:                       # clean flags by default (grid_frame filters)
                    row["TESS_flags"] = np.zeros(int(keep.sum()), dtype=int)
                    row["TGLC_flags"] = np.zeros(int(keep.sum()), dtype=int)
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


def _grid_one_star(tess=None, tglc=None, flux=None, n=300, t0=1683.0, t1=1710.0):
    """Grid a single synthetic star through grid_frame (shared arm)."""
    times = np.linspace(t0, t1, n)
    flux = np.full(n, 1000.0) if flux is None else np.asarray(flux, dtype=float)
    tess = np.zeros(n, dtype=int) if tess is None else np.asarray(tess, dtype=int)
    tglc = np.zeros(n, dtype=int) if tglc is None else np.asarray(tglc, dtype=int)
    df = pd.DataFrame([{"TIC": "T0", "sector": 14, "camera": 1, "ccd": 1,
                        "time": times, "flux": flux,
                        "TESS_flags": tess, "TGLC_flags": tglc}])
    X, M = grid_frame(df, "shared", (t0, t1))
    return times, X[0], M[0], (t0, t1)


def _bin_of(time_value, t0, t1):
    return int(shared_grid_bin(np.array([time_value]), t0, t1)[0])


def _assert_flagged_cadence_removed(tess_bit=0, tglc_val=0, idx=150):
    tess = np.zeros(300, dtype=int); tglc = np.zeros(300, dtype=int)
    tess[idx] = tess_bit; tglc[idx] = tglc_val
    times, _, M, (t0, t1) = _grid_one_star(tess=tess, tglc=tglc)
    assert M[_bin_of(times[idx], t0, t1)] == 0.0        # flagged cadence -> empty bin


def test_grid_frame_requires_flag_columns():
    df = synthetic_frame().drop(columns=["TESS_flags", "TGLC_flags"])
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    try:
        grid_frame(df, "shared", (t0, t1))
    except RuntimeError:
        return
    raise AssertionError("grid_frame must fail loudly without TESS_flags/TGLC_flags")


def test_flag_32_momentum_dump_removed():
    _assert_flagged_cadence_removed(tess_bit=32)


def test_all_five_tess_flags_removed():
    for bit in (1, 4, 16, 32, 16384):
        _assert_flagged_cadence_removed(tess_bit=bit)


def test_tglc_nonzero_removed():
    _assert_flagged_cadence_removed(tglc_val=1)
    _assert_flagged_cadence_removed(tglc_val=5)


def test_clean_cadences_remain():
    times, _, M, (t0, t1) = _grid_one_star()            # all flags zero
    assert M[_bin_of(times[150], t0, t1)] == 1.0
    assert M.sum() > 0


def test_removed_cadence_does_not_affect_normalization():
    flux = 1000.0 + 30.0 * np.sin(np.linspace(0.0, 10.0, 300))
    flux[150] = 1e9                                      # extreme outlier ...
    tess = np.zeros(300, dtype=int); tess[150] = 32      # ... that is flagged
    times, X, M, (t0, t1) = _grid_one_star(tess=tess, flux=flux)
    assert M[_bin_of(times[150], t0, t1)] == 0.0         # flagged bin stays empty
    assert np.abs(X).max() < 100.0                       # 1e9 never enters norm or grid


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
