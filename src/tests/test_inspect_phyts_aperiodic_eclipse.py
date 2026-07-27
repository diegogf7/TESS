import numpy as np
import torch

from src.instrument_v2.inspect_phyts_aperiodic_eclipse import pick_first_by_tic, predict_arm

from src.instrument_v2.eval_phyts_instrument_ab import matched_split, instrument_cleaned_curve

from src.data.data import CLASSES, CLASS_TO_IDX
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


AP = CLASS_TO_IDX["APERIODIC"]
EC = CLASS_TO_IDX["ECLIPSE"]

def test_pick_first_by_tic_selects_exactly_one_sorted():

    test_tics = np.array(["T9", "T2", "T5", "T2", "T7"])
    test_y = np.array([AP, AP, EC, EC, AP])
    ap = pick_first_by_tic(test_tics, test_y, AP)

    ec = pick_first_by_tic(test_tics, test_y, EC)
    assert ap == 1 and ec == 3
    assert test_y[ap] == AP and test_y[ec] == EC
    assert isinstance(ap, int) and isinstance(ec, int)
    assert pick_first_by_tic(test_tics, np.full(5, AP), EC) is None

def test_both_arms_use_same_deterministic_split():

    tics = np.array([f"T{i // 3}" for i in range(60)])
    y = np.array([i % 4 for i in range(60)])

    tr1, te1 = matched_split(tics, y)
    tr2, te2 = matched_split(tics, y)

    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)


def test_predict_arm_uses_class_mapping():

    rng = np.random.default_rng(0)
    y = np.array([i % len(CLASSES) for i in range(80)])
    lat = rng.normal(size =(80, 16)) + y[:, None]
    train_index = np.arange(60); test_index = np.arange(60, 80)
    prediction = predict_arm(lat, y, train_index, test_index)

    assert prediction.shape == (20,)
    assert prediction.min() >= 0 and prediction.max() < len(CLASSES)

    _ = [CLASSES[p] for p in prediction]



def test_model_hashes_unchanged():

    torch.manual_seed(0)
    instance = FixedTeacherInstrumentJEPA(n_tokens = 16, token_dim = 16, d_model = 32, n_layers = 1, readout = "mean", predictor_type = "mlp").eval()

    for p in instance.parameters():
        p.requires_grad_(False)
    decoder = build_decoder(1024)
    for p in decoder.parameters():
        p.requires_grad_(False)
    
    before = (state_hash(instance.teacher), state_hash(instance.student), state_hash(instance.predictor), state_hash(decoder))

    time = np.linspace(1683.0, 1710.0, 300)
    instrument_cleaned_curve(time, np.random.default_rng(1).normal(1000, 30, 300), instance, decoder, 1683.0, 1710.0)

    after = (state_hash(instance.teacher), state_hash(instance.student), state_hash(instance.predictor), state_hash(decoder))

    assert after == before

if __name__ == "__main__":

    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print(f"ALL {len(tests)} INSPECT-APERIODIC-ECLIPSE TESTS PASSED")