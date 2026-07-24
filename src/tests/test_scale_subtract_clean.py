# All this code is from Claude
"""Contracts for the scale-and-subtract cleaner. Synthetic only; no cluster,
checkpoints, or test TICs.

Run: python -m src.tests.test_scale_subtract_clean
"""

import numpy as np
import torch

from src.instrument_v2.scale_subtract_clean import (
    BAD_TESS_MASK,
    cadence_good,
    decode,
    robust_amplitude,
    scale_and_subtract,
)
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


def test_mask_value_is_16437():
    assert BAD_TESS_MASK == 16437 == (1 | 4 | 16 | 32 | 16384)


def test_momentum_dump_flag_32_rejected():
    assert not cadence_good([32], [0])[0]


def test_all_five_tess_flags_rejected():
    for f in (1, 4, 16, 32, 16384):
        assert not cadence_good([f], [0])[0]
    assert not cadence_good([1 | 32 | 16384], [0])[0]        # combined still rejected


def test_clean_flags_retained():
    assert cadence_good([0], [0])[0]
    for f in (2, 8, 64, 128):                                 # bits NOT in the bad mask
        assert (f & BAD_TESS_MASK) == 0
        assert cadence_good([f], [0])[0]


def test_tglc_nonzero_rejected():
    assert not cadence_good([0], [1])[0]
    assert not cadence_good([0], [7])[0]
    assert cadence_good([0], [0])[0]


def test_synthetic_scale_recovered_and_nonnegative():
    rng = np.random.default_rng(1)
    dec = rng.normal(size=500)
    dec_c = dec - np.median(dec)
    assert abs(robust_amplitude(2.5 * dec_c, dec_c) - 2.5) < 1e-3
    assert robust_amplitude(-4.0 * dec_c, dec_c) == 0.0       # a >= 0 clamp


def test_rejected_cadences_never_enter_fit():
    rng = np.random.default_rng(0)
    dec = rng.normal(size=1024)
    raw = 3.0 * (dec - np.median(dec))
    mask = np.ones(1024, dtype=bool)
    bad = np.arange(400, 470)                                 # a gap -> new segment boundary
    raw[bad], dec[bad], mask[bad] = 1e6, -1e6, False          # garbage, but rejected
    instrument, cleaned, amps = scale_and_subtract(raw, dec, mask)
    assert amps and abs(amps[0] - 3.0) < 0.05                 # fit unaffected by the garbage
    assert np.allclose(instrument[bad], 0.0)                  # rejected bins never fit


def test_output_length_1024_and_hashes_unchanged():
    torch.manual_seed(0)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32,
                                       n_layers=1, readout="mean", predictor_type="mlp")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    dec = build_decoder(1024)
    before = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    template = decode(model, dec, np.random.randn(64).astype(np.float32),
                      np.ones(64, dtype=np.float32))
    assert template.shape == (1024,)
    after = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    assert after == before                                    # frozen nets untouched


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} SCALE-SUBTRACT-CLEAN TESTS PASSED")
