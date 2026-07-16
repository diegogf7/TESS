# All this code is from Claude
"""Unit tests for src/instrument_v2/diagnose_chip_common_signal.py.

Pure synthetic data -- no cluster files, no torch, no matplotlib.
Run: python -m src.tests.test_chip_common_signal   (or pytest)
"""

import os
import tempfile

import numpy as np

from src.instrument_v2.diagnose_chip_common_signal import (
    apply_quality_mask,
    build_exact_cadence,
    chip_index,
    chip_name,
    fit_chip_bases,
    load_or_create_split,
    make_split,
    paired_bootstrap,
    permute_labels,
    predict_chips,
    recon_error,
    shuffle_within_curves,
)


def test_cadence_maps_to_identical_index():
    """Same cadence_num -> same tensor index for every star, regardless of
    each star's own coverage."""
    cad_a = np.array([100, 101, 102, 105, 110])
    cad_b = np.array([102, 103, 105, 110, 111, 112])   # different start/coverage
    flux_a = np.arange(5, dtype=float) + 1.0
    flux_b = np.arange(6, dtype=float) + 100.0
    X, M, first = build_exact_cadence([cad_a, cad_b], [flux_a, flux_b])

    assert first == 100
    for cad_shared in (102, 105, 110):
        idx = cad_shared - first                      # the one shared rule
        ia = np.flatnonzero(cad_a == cad_shared)[0]
        ib = np.flatnonzero(cad_b == cad_shared)[0]
        assert X[0, idx] == flux_a[ia], "star A value not at cadence index"
        assert X[1, idx] == flux_b[ib], "star B value not at cadence index"
        assert M[0, idx] == 1.0 and M[1, idx] == 1.0
    # cadence observed by only one star: observed flag differs, index identical
    assert M[0, 100 - first] == 1.0 and M[1, 100 - first] == 0.0
    # grid is padded to a multiple of 16
    assert X.shape[1] % 16 == 0


def test_split_tics_disjoint_and_deterministic():
    tics = [f"TIC{i}" for i in range(100)] * 2        # duplicates must not matter
    train1, test1 = make_split(tics, seed=42)
    train2, test2 = make_split(tics, seed=42)
    assert train1 == train2 and test1 == test2, "split not deterministic"
    assert not (train1 & test1), "train/test TICs overlap"
    assert len(train1) + len(test1) == 100
    assert len(test1) == 20                            # 80/20 on unique TICs
    other_train, _ = make_split(tics, seed=43)
    assert other_train != train1, "seed has no effect -- suspicious"


def test_bases_never_see_test_data():
    """Perturbing test rows must leave the fitted chip bases bit-identical."""
    rng = np.random.default_rng(0)
    n, d = 40, 64
    X = rng.normal(size=(n, d)).astype(np.float32)
    M = np.ones_like(X)
    chips = np.repeat(np.arange(4), 10)
    is_train = np.arange(n) % 5 != 0                   # 32 train / 8 test

    bases1, _ = fit_chip_bases(X[is_train], M[is_train], chips[is_train], k_max=4)
    X_perturbed = X.copy()
    X_perturbed[~is_train] *= 1e6                      # nuke the test rows
    bases2, _ = fit_chip_bases(X_perturbed[is_train], M[is_train], chips[is_train], k_max=4)

    assert bases1.keys() == bases2.keys()
    for chip in bases1:
        np.testing.assert_array_equal(bases1[chip][0], bases2[chip][0])
        np.testing.assert_array_equal(bases1[chip][1], bases2[chip][1])


def test_recon_error_ignores_unobserved():
    rng = np.random.default_rng(1)
    d = 64
    mean = np.zeros(d)
    components = rng.normal(size=(4, d))
    m = (rng.random(d) > 0.3).astype(np.float32)       # ~70% observed
    x = rng.normal(size=d)

    e1 = recon_error(x, m, mean, components, k=4)
    x_garbage = x.copy()
    x_garbage[m == 0] = 1e9                            # garbage where unobserved
    e2 = recon_error(x_garbage, m, mean, components, k=4)
    assert e1 == e2, "reconstruction error depends on unobserved values"

    # too few observed points for K -> NaN (star counted as skipped), not a crash
    m_tiny = np.zeros(d, dtype=np.float32)
    m_tiny[:3] = 1.0
    assert np.isnan(recon_error(x, m_tiny, mean, components, k=4))


def test_chip_labels_map_consistently():
    seen = set()
    for camera in range(1, 5):
        for ccd in range(1, 5):
            idx = chip_index(camera, ccd)
            assert idx == (camera - 1) * 4 + (ccd - 1)
            assert 0 <= idx <= 15
            assert chip_name(idx) == f"cam{camera}-ccd{ccd}"
            seen.add(idx)
    assert seen == set(range(16)), "mapping is not a bijection onto 0..15"
    for bad in ((0, 1), (5, 1), (1, 0), (1, 5)):
        try:
            chip_index(*bad)
            assert False, f"chip_index accepted out-of-range {bad}"
        except ValueError:
            pass


def test_shuffle_control_preserves_mask_and_values():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 32)).astype(np.float32)
    M = (rng.random((5, 32)) > 0.4).astype(np.float32)
    X *= M
    Xs = shuffle_within_curves(X, M, seed=7)
    for i in range(5):
        obs = M[i] > 0
        assert np.array_equal(np.sort(Xs[i, obs]), np.sort(X[i, obs])), \
            "shuffle changed the multiset of observed values"
        assert np.all(Xs[i, ~obs] == X[i, ~obs]), "shuffle touched unobserved entries"


def test_end_to_end_recovers_planted_common_mode():
    """Stars sharing a chip-specific temporal pattern must be classified
    correctly by the basis pipeline (sanity of the whole loop)."""
    rng = np.random.default_rng(3)
    n_per, d = 30, 128
    patterns = rng.normal(size=(4, d)) * 2.0
    X, chips = [], []
    for chip in range(4):
        for _ in range(n_per):
            X.append(patterns[chip] * rng.uniform(0.5, 1.5) + rng.normal(size=d) * 0.3)
            chips.append(chip)
    X = np.asarray(X, dtype=np.float32)
    chips = np.asarray(chips)
    M = (rng.random(X.shape) > 0.15).astype(np.float32)
    X *= M
    is_train = np.arange(len(X)) % 5 != 0
    bases, skipped = fit_chip_bases(X[is_train], M[is_train], chips[is_train], k_max=4)
    assert set(bases) == {0, 1, 2, 3}
    assert all(n == 0 for _, n in skipped), "populated chip wrongly skipped"
    pred, _, n_unpred = predict_chips(X[~is_train], M[~is_train], bases, k=1)
    assert n_unpred == 0
    accuracy = float(np.mean(pred == chips[~is_train]))
    assert accuracy > 0.9, f"planted common mode not recovered (acc {accuracy})"


def test_k0_uses_chip_mean_only():
    """K=0 must classify by the chip's masked mean curve, no components."""
    d = 32
    mean_a, mean_b = np.full(d, 1.0), np.full(d, -1.0)
    comps = np.random.default_rng(4).normal(size=(4, d))
    m = np.ones(d, dtype=np.float32)
    x = np.full(d, 0.9)                               # much closer to mean_a
    e_a = recon_error(x, m, mean_a, comps, k=0)
    e_b = recon_error(x, m, mean_b, comps, k=0)
    assert abs(e_a - np.mean((x - mean_a) ** 2)) < 1e-12, "K=0 error is not masked MSE to mean"
    assert e_a < e_b
    # K=0 must ignore the components entirely
    e_a2 = recon_error(x, m, mean_a, comps * 100, k=0)
    assert e_a == e_a2, "K=0 depends on PCA components"


def test_quality_mask_keeps_only_clean_cadences():
    time = np.arange(6, dtype=float)
    flux = np.arange(6, dtype=float) * 10
    cad = np.arange(100, 106)
    tess = np.array([0, 0, 1, 0, 0, 0])
    tglc = np.array([0, 0, 0, 0, 2, 0])
    t, f, c = apply_quality_mask(time, flux, cad, [tess, tglc])
    assert list(c) == [100, 101, 103, 105], "wrong cadences kept"
    assert list(f) == [0.0, 10.0, 30.0, 50.0]
    assert len(t) == 4


def test_permuted_labels_preserve_multiset():
    y = np.repeat(np.arange(4), 10)
    yp = permute_labels(y, seed=0)
    assert not np.array_equal(y, yp), "permutation did nothing"
    assert np.array_equal(np.sort(y), np.sort(yp)), "permutation changed label counts"
    assert np.array_equal(yp, permute_labels(y, seed=0)), "not deterministic"


def test_paired_bootstrap_direction():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, size=400)
    good = y.copy()
    wrong = rng.integers(0, 4, size=400)
    good[rng.random(400) < 0.2] = rng.integers(0, 4, size=int((rng.random(400) < 0.2).sum()) or 1)[0]
    d, lo, hi, p = paired_bootstrap(y, good, wrong, n_boot=200, seed=0)
    assert d > 0 and p < 0.05, f"clearly-better predictor not detected (d={d}, p={p})"
    d0, _, _, p0 = paired_bootstrap(y, wrong, wrong, n_boot=200, seed=0)
    assert d0 == 0.0, "identical predictors must have zero diff"


def test_split_saved_and_reloaded_identically():
    tics = [f"TIC{i}" for i in range(50)]
    with tempfile.TemporaryDirectory() as tmp:
        train1, test1 = load_or_create_split(tics, tmp, seed=42)
        assert os.path.exists(os.path.join(tmp, "split_train_tics.txt"))
        train2, test2 = load_or_create_split(tics, tmp, seed=999)  # seed ignored on reload
        assert train1 == train2 and test1 == test2, "reloaded split differs"
        try:
            load_or_create_split(tics + ["TIC_NEW"], tmp, seed=42)
            assert False, "mismatched TIC set accepted"
        except RuntimeError:
            pass


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
