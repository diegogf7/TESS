import os
import tempfile
import numpy as np
import pandas as pd
import torch

from src.instrument_v2.eval_phyts_instrument_ab import(

    assert_classes_present,
    instrument_cleaned_curve,
    matched_split,
    ordered_hash,
    physics_grid,
)
from src.data.data import CLASSES, DualEvalDataset
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad

from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import grid_curve_shared

T0, T1 = 1683, 1710.0

def _tiny_instrument():

    torch.manual_seed(0)
    m = FixedTeacherInstrumentJEPA(n_tokens = 16, token_dim = 16, d_model = 32, n_layers = 1, readout = "mean", predictor_type = "mlp")

    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    
    return m

def test_arm_a_matches_dualeval_dataset():

    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):

        time = np.linspace(1683.0, 1710.0, 400)
        rows.append({"TIC": f"T{i}", "sector": 14, "label": CLASSES[i % len(CLASSES)], "time": time, "flux": rng.normal(1000, 30, 400)})

    path = os.path.join(tempfile.mkdtemp(), "phyts.parquet")
    pd.DataFrame(rows).to_parquet(path)

    ds = DualEvalDataset(path, 1024)

    for i in range(len(ds)):
        gf, ob, _, _ = ds[i]

        a_x, a_m = physics_grid(ds.df["time"].iloc[i], ds.df["flux"].iloc[i])
        assert np.allclose(a_x, gf.numpy(), atol = 1e-5)
        assert np.allclose(a_m, ob.numpy())

def test_instrument_clean_subtraction_no_scale_fit():

    model, decoder = _tiny_instrument(), build_decoder(1024)

    time = np.linspace(T0, T1, 300)
    flux = 1000.0 + 40.0 * np.sin(np.linspace(0, 12, 300))

    ct, cf = instrument_cleaned_curve(time, flux, model, decoder, T0, T1)
    assert np.isfinite(ct).all() and np.isfinite(cf).all()


    normed, med, mad = normalize_median_mad(flux)
    scale = 1.4826 * mad

    X, M = grid_curve_shared(time, normed, T0, T1, 1024)

    with torch.no_grad():
        z = model.encode(torch.tensor(X, dtype = torch.float32)[None], torch.tensor(M, dtype = torch.float32)[None], view = "predicted")
        decoded = decoder(z).squeeze(0).numpy()

    valid = M >0
    expected = ((X - decoded) * scale + med)[valid]
    assert np.allclose(cf, expected, atol = 1e-4)

def test_matched_split_is_tic_disjoin_and_deterministic():

    tics = np.array([f"T{i // 3}" for i in range(60)])
    y= np.array([i % 4 for i in range(60)])
    tr, te = matched_split(tics, y)

    assert not (set(tics[tr]) & set(tics[te]))

    tr2, te2 = matched_split(tics, y)

    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)


def test_assert_classes_present_aborts_when_missing():

    y = np.array([0, 0, 0, 1])
    raised = False

    try:
        assert_classes_present(y, np.array([0, 1, 2]), np.array([3]))
    except RuntimeError:
        raised = True

    assert raised
    assert_classes_present(np.array([0, 1, 0, 1]), np.array([0, 1]), np.array([2, 3]))


def test_ordered_hash_deterministic_and_order_sensitive():

    tics = np.array(["A", "B", "C"]); y = np.array([0, 1, 2])
    assert ordered_hash(tics, y) == ordered_hash(tics, y)
    assert ordered_hash(tics, y) != ordered_hash(tics[::-1], y[::-1])


def test_model_hashes_unchanged_after_cleaning():

    model, decoder = _tiny_instrument(), build_decoder(1024)
    for p in decoder.parameters():

        p.requires_grad_(False)
    before = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor), state_hash(decoder))

    time = np.linspace(T0, T1, 300)
    instrument_cleaned_curve(time, np.random.default_rng(1).normal(1000, 30, 300), model, decoder, T0, T1)

    after = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor), state_hash(decoder))

    assert after == before

if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]

    for test in tests:
        test()
        print(f"Pass {test.__name__}")
    print(f"All {len(tests)} PHYTS-INSTRUMENT-AB TESTS PASSED")