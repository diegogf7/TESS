# All this code is from Claude
"""Tests for the cadence-aligned dataset. Plain asserts, no pytest needed:

    python -m src.tests.test_cadence_dataset

Verifies the core alignment guarantee: a shared FFI cadence lands at the
SAME tensor index for every star, quality flags mask the right cadences,
and the grid pads to a token-friendly length.
"""

import numpy as np
import pandas as pd

from src.data.cadence_dataset import (CadenceAlignedDataset, place_on_grid,
                                      sector_cadence_grid)


def synthetic_df():
    # Three stars, sector 7, overlapping cadence ranges.
    #   star 1: cadences 100..109
    #   star 2: cadences 105..114 (overlaps 105..109 with star 1)
    #   star 3: cadences 100..114, cadence 107 flagged bad
    rows = []
    for gaia, (c0, c1) in [(1, (100, 109)), (2, (105, 114)), (3, (100, 114))]:
        cadences = np.arange(c0, c1 + 1)
        rows.append({
            "GAIADR3": gaia,
            "sector": 7,
            "time": cadences * 0.0208,
            "flux": 1000.0 + cadences.astype(float),  # value encodes cadence -> checkable
            "cadence_num": cadences,
            "TESS_flags": np.where((gaia == 3) & (cadences == 107), 1, 0),
            "TGLC_flags": np.zeros(len(cadences), dtype=int),
        })
    return pd.DataFrame(rows)


def test_grid_construction():
    first, length = sector_cadence_grid([np.arange(100, 110), np.arange(105, 115)],
                                        pad_to_multiple=16)
    assert first == 100, first
    assert length == 16, length  # span 15, padded up to 16
    first, length = sector_cadence_grid([np.arange(0, 32)], pad_to_multiple=16)
    assert length == 32, length  # already a multiple: no padding


def test_place_on_grid_rejects_out_of_range():
    try:
        place_on_grid(np.ones(3), np.array([5, 6, 99]), first_cadence=5, grid_length=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for cadence outside grid")


def test_shared_cadence_same_index():
    ds = CadenceAlignedDataset(synthetic_df(), sector=7)
    idx_105 = ds.index_of_cadence(105)
    assert idx_105 == 5, idx_105  # first cadence 100 -> index 5

    tensors = [ds[i] for i in range(len(ds))]
    for flux, observed, _ in tensors:
        assert flux.shape[0] == ds.grid_length == 16, ds.grid_length

    # every star that observed cadence 105 has it at the SAME index,
    # carrying that star's own value (1000 + 105 before normalization).
    for star_row, (flux, observed, _) in zip(range(3), tensors):
        assert observed[idx_105] == 1.0, f"star {star_row + 1} missing cadence 105"
    # star 1 did not observe cadence 112; star 2 did.
    idx_112 = ds.index_of_cadence(112)
    assert tensors[0][1][idx_112] == 0.0
    assert tensors[1][1][idx_112] == 1.0
    # no interpolation: unobserved grid slots hold exactly 0 flux.
    assert tensors[0][0][idx_112] == 0.0


def test_quality_mask():
    ds = CadenceAlignedDataset(synthetic_df(), sector=7)
    idx_107 = ds.index_of_cadence(107)
    _, observed3, quality3 = ds[2]  # star 3 has TESS_flags=1 at cadence 107
    assert observed3[idx_107] == 1.0   # observed...
    assert quality3[idx_107] == 0.0    # ...but flagged bad
    _, _, quality1 = ds[0]             # star 1 is clean at 107
    assert quality1[idx_107] == 1.0


def test_normalization_uses_only_observed_points():
    ds = CadenceAlignedDataset(synthetic_df(), sector=7)
    flux, observed, _ = ds[0]
    observed_values = flux[observed.bool()]
    # normalize() centres relative flux at ~0; padding zeros must not shift it.
    assert abs(float(observed_values.median())) < 0.1


if __name__ == "__main__":
    test_grid_construction()
    test_place_on_grid_rejects_out_of_range()
    test_shared_cadence_same_index()
    test_quality_mask()
    test_normalization_uses_only_observed_points()
    print("ALL CADENCE DATASET TESTS PASSED")
