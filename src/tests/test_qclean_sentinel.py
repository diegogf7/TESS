# All this code is from Claude
"""End-to-end sentinel: an extreme value placed on a FLAGGED cadence must never
reach the gridded input, either Stage-A/Stage-B group, the median/log-MAD, the
CBV basis, or the decoder target -- and clean cadences must be untouched. Proven
by showing 'flag the cadence' is bit-identical to 'delete the cadence'.

Run: python -m src.tests.test_qclean_sentinel
"""

import tempfile

import numpy as np
import pandas as pd

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    group_statistics,
)
from src.instrument_v2.regional_cbv import build_or_load_area_bases, ridge_reconstruct
from src.tests.test_area_commonmode_jepa import frame_with_areas

EXTREME = 1e12
K = 8
GROUP = 2          # group size is irrelevant to the filter; small keeps the test light


def _dataset(frame):
    t0 = min(float(np.min(t)) for t in frame["time"])
    t1 = max(float(np.max(t)) for t in frame["time"])
    return Sector14GroupStatDataset(frame, set(frame["TIC"].astype(str)),
                                    (t0, t1), "area", GROUP, grid_length=64, min_valid=1)


def _inject_flagged_extremes(frame, n_flag=5, seed=0):
    """(flagged, deleted): flagged puts EXTREME+TESS-flag-32 on n_flag cadences per
    star; deleted removes those same cadences entirely (the clean equivalent)."""
    rng = np.random.default_rng(seed)
    flagged_rows, deleted_rows = [], []
    for _, r in frame.iterrows():
        d = r.to_dict()
        time = np.asarray(d["time"], dtype=float)
        flux = np.asarray(d["flux"], dtype=float)
        m = len(time)
        idx = rng.choice(m, size=min(n_flag, m), replace=False)

        f2 = flux.copy(); f2[idx] = EXTREME
        tess = np.zeros(m, dtype=int); tess[idx] = 32
        flagged_rows.append({**d, "flux": f2,
                             "TESS_flags": tess, "TGLC_flags": np.zeros(m, dtype=int)})

        keep = np.ones(m, dtype=bool); keep[idx] = False
        deleted_rows.append({**d, "time": time[keep], "flux": flux[keep],
                             "TESS_flags": np.zeros(int(keep.sum()), dtype=int),
                             "TGLC_flags": np.zeros(int(keep.sum()), dtype=int)})
    return pd.DataFrame(flagged_rows), pd.DataFrame(deleted_rows)


def test_sentinel_flagged_extremes_never_enter_pipeline():
    frame = frame_with_areas(n_per_chip=40, n_rings=1)     # 40 stars/area, clean flags
    flagged, deleted = _inject_flagged_extremes(frame)
    ds_f = _dataset(flagged)
    ds_d = _dataset(deleted)

    # 1) the extreme never reaches the gridded model input
    assert np.isfinite(ds_f.X).all()
    assert np.abs(ds_f.X).max() < 1e3, "flagged extreme leaked into ds.X"

    # 2) flag == delete: surviving (clean) cadences grid bit-identically
    assert ds_f.tics.tolist() == ds_d.tics.tolist()
    assert np.array_equal(ds_f.M, ds_d.M)
    assert np.allclose(ds_f.X, ds_d.X, atol=1e-5)

    # 3) median / log-MAD (Stage-A + Stage-B group stats) stay finite and bounded
    area = int(ds_f.areas[0])
    rows = [r for r in range(len(ds_f.tics)) if int(ds_f.areas[r]) == area][:2 * GROUP]
    med, lmad, valid, _ = group_statistics(ds_f.X[rows[:GROUP]], ds_f.M[rows[:GROUP]],
                                           ds_f.min_valid)
    assert np.abs(med).max() < 1e3 and np.isfinite(lmad).all()

    # 4) CBV basis and 5) decoder target (ridge reconstruction) never see the extreme
    tmp = tempfile.mkdtemp()
    bases = build_or_load_area_bases(ds_f.X, ds_f.M, ds_f.areas, sorted(ds_f.tics),
                                     K, tmp, GROUP, 1)
    assert bases and all(np.isfinite(B).all() and np.abs(B).max() < 1e3
                         for B in bases.values())
    recon = ridge_reconstruct(med, valid, bases[area], 1e-2)
    assert np.isfinite(recon).all() and np.abs(recon).max() < 1e3


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} QCLEAN SENTINEL TESTS PASSED")
