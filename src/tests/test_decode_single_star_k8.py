# All this code is from Claude
"""Smoke test for the single-star K=8 decode. Synthetic only; no cluster,
no checkpoints, no test TICs.

Run: python -m src.tests.test_decode_single_star_k8
"""

import numpy as np
import torch

from src.instrument_v2.decode_single_star_k8 import (
    build_decoder,
    masked_metrics,
    masked_smooth_l1,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


def test_decoder_output_length_1024():
    out = build_decoder()(torch.randn(3, 16, 16))
    assert out.shape == (3, 1024)


def test_weight_decoder_outputs_8_and_reconstructs_1024():
    w = build_decoder(8)(torch.randn(3, 16, 16))
    assert w.shape == (3, 8)                       # decoder output [batch, 8]
    B = torch.randn(1024, 8)                       # area basis
    curve = w @ B.T
    assert curve.shape == (3, 1024)                # reconstructed curve [batch, 1024]


def test_masked_loss_ignores_gaps():
    torch.manual_seed(0)
    p, t = torch.randn(2, 1024), torch.randn(2, 1024)
    m = torch.ones(2, 1024); m[:, 100:300] = 0.0
    base = masked_smooth_l1(p, t, m)
    poisoned = p.clone(); poisoned[:, 100:300] = 999.0
    assert torch.allclose(base, masked_smooth_l1(poisoned, t, m))


def test_masked_metrics_perfect_on_identity():
    x = np.random.default_rng(0).normal(size=1024).astype(np.float32)
    m = np.ones(1024); m[:200] = 0.0
    c, rmse, r2 = masked_metrics(x, x, m)
    assert abs(c - 1) < 1e-6 and rmse < 1e-6 and abs(r2 - 1) < 1e-6


def test_only_decoder_gets_grads_and_frozen_nets_unchanged():
    torch.manual_seed(0)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32,
                                       n_layers=1, readout="mean", predictor_type="mlp")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    dec = build_decoder()
    before = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    raw, mask = torch.randn(2, 64), torch.ones(2, 64)
    predicted_latent = model.encode(raw, mask, view="predicted")   # frozen S4D + predictor
    pred = dec(predicted_latent)
    assert pred.shape == (2, 1024)
    masked_smooth_l1(pred, torch.randn(2, 1024), torch.ones(2, 1024)).backward()
    assert all(p.grad is None for p in model.parameters())         # frozen: no grads
    assert any(p.grad is not None for p in dec.parameters())       # decoder trains
    after = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    assert after == before                                          # bit-identical


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} DECODE-K8 SMOKE TESTS PASSED")
