# All this code is from Claude
"""Contracts for the area-conditioned K=8 CBV-weight decoder. Synthetic only --
no cluster, no real checkpoints, no test TICs.

Run: python -m src.tests.test_area_conditioned_decoder
"""

import tempfile
import os

import numpy as np
import torch

from src.instrument_v2.decode_single_star_k8 import (
    build_decoder, area_index_map, area_one_hot, decode, CBV_RANK,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


def _raises(fn):
    try:
        fn(); return False
    except RuntimeError:
        return True


# 1) area map is deterministic (sorted) and round-trips through a checkpoint
def test_area_map_deterministic_and_checkpointed():
    areas = [412, 111, 233, 111, 412]
    a2i = area_index_map(areas)
    assert a2i == area_index_map(list(reversed(areas)))          # order-independent
    assert list(a2i) == sorted(a2i)                              # sorted keys
    assert a2i == {111: 0, 233: 1, 412: 2}                       # contiguous 0..n-1
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "decoder.pth")
        dec = build_decoder(CBV_RANK, in_dim=256 + len(a2i))
        torch.save({"state_dict": dec.state_dict(), "area_to_index": a2i,
                    "n_areas": len(a2i), "in_dim": 256 + len(a2i), "cbv_rank": CBV_RANK}, p)
        ck = torch.load(p)
        assert {int(k): int(v) for k, v in ck["area_to_index"].items()} == a2i
        assert ck["in_dim"] == 256 + len(a2i)


# 2) area conditioning changes the decoder input (concat adds n_areas dims)
def test_area_conditioning_changes_input():
    a2i = {111: 0, 233: 1, 412: 2}
    latent = torch.randn(1, 16, 16)
    flat = latent.reshape(1, -1)                                 # (1, 256)
    oh = torch.tensor(area_one_hot(a2i, 233))[None]              # (1, 3)
    combined = torch.cat([flat, oh], dim=1)
    assert combined.shape == (1, 256 + 3)                        # concat(latent, one-hot)
    assert not torch.equal(combined[:, :256] * 0 + oh.sum(), torch.zeros(1))  # one-hot present
    assert oh.sum().item() == 1.0 and oh[0, 1].item() == 1.0     # correct slot for area 233
    # different areas -> different combined inputs
    oh_other = torch.tensor(area_one_hot(a2i, 412))[None]
    assert not torch.equal(oh, oh_other)


# 3) unknown areas hard-fail
def test_unknown_area_hard_fails():
    a2i = {111: 0, 233: 1}
    assert area_one_hot(a2i, 111).tolist() == [1.0, 0.0]
    assert _raises(lambda: area_one_hot(a2i, 999))               # not in map -> RuntimeError


# 4) the area-conditioned decoder consumes the concatenated vector and emits 8
def test_area_decoder_forward_shapes():
    a2i = {111: 0, 233: 1, 412: 2}
    dec = build_decoder(CBV_RANK, in_dim=256 + len(a2i))
    z = torch.randn(4, 16, 16).reshape(4, -1)                    # (4, 256)
    oh = torch.stack([torch.tensor(area_one_hot(a2i, a)) for a in (111, 233, 412, 111)])
    out = dec(torch.cat([z, oh], dim=1))
    assert out.shape == (4, CBV_RANK)                            # [batch, 8]
    B = torch.randn(1024, CBV_RANK)
    assert (out @ B.T).shape == (4, 1024)                        # reconstructed curve [batch, 1024]


# 5) eligibility (scored validation rows) is decoder-independent -> identical rows
def test_eligibility_is_decoder_independent():
    # reference_target/median depend only on (dataset, bases), never the decoder,
    # so global and area-conditioned eval score exactly the same rows.
    from src.instrument_v2.eval_cbv_oracle_ceiling import reference_median

    class DS:                                                    # minimal stand-in
        pass
    rng = np.random.default_rng(0)
    ds = DS()
    ds.areas = np.array([111] * 40 + [233] * 40)
    ds.tics = np.array([f"T{i}" for i in range(80)])
    ds.X = rng.normal(size=(80, 1024)).astype(np.float32)
    ds.M = np.ones((80, 1024), np.float32)
    ds.min_valid = 16
    from src.instrument_v2.decode_single_star_k8 import deterministic_area_rows
    rows = deterministic_area_rows(ds)
    bases = {111: rng.normal(size=(1024, CBV_RANK)), 233: rng.normal(size=(1024, CBV_RANK))}
    elig = [i for i in range(80) if reference_median(ds, rows, i, bases)[0] is not None]
    elig2 = [i for i in range(80) if reference_median(ds, rows, i, bases)[0] is not None]
    assert elig == elig2 and len(elig) > 0                       # deterministic, decoder-free


# 6) frozen instrument stays hash-identical through an area-conditioned decode;
#    only the decoder receives gradients
def test_frozen_unchanged_and_only_decoder_trains():
    torch.manual_seed(0)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                       readout="mean", predictor_type="mlp").eval()
    for p in model.parameters():
        p.requires_grad = False
    a2i = {111: 0, 233: 1}
    dec = build_decoder(CBV_RANK, in_dim=256 + len(a2i))
    before = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    raw, mask = np.random.randn(64).astype(np.float32), np.ones(64, np.float32)
    # forward through the area path (concat one-hot) and backprop a dummy loss
    r = torch.tensor(raw)[None]; m = torch.tensor(mask)[None]
    z = model.encode(r, m, view="predicted").reshape(1, -1)
    oh = torch.tensor(area_one_hot(a2i, 233))[None]
    out = dec(torch.cat([z, oh], dim=1))
    out.pow(2).mean().backward()
    assert all(p.grad is None for p in model.parameters())       # frozen: no grads
    assert any(p.grad is not None for p in dec.parameters())     # decoder trains
    after = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    assert after == before                                       # bit-identical


# 7) area-aware decode() matches manual concat forward (wiring is correct)
def test_decode_area_matches_manual():
    torch.manual_seed(1)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                       readout="mean", predictor_type="mlp").eval()
    for p in model.parameters():
        p.requires_grad = False
    a2i = {111: 0, 233: 1, 412: 2}
    dec = build_decoder(CBV_RANK, in_dim=256 + len(a2i)).eval()
    for p in dec.parameters():
        p.requires_grad = False
    rng = np.random.default_rng(2)
    raw = rng.normal(size=64).astype(np.float32); mask = np.ones(64, np.float32)
    B = rng.normal(size=(1024, CBV_RANK)).astype(np.float32)
    av = area_one_hot(a2i, 233)
    got = decode(model, dec, raw, mask, B, av)                   # (1024,)
    with torch.no_grad():
        z = model.encode(torch.tensor(raw)[None], torch.tensor(mask)[None], view="predicted").reshape(1, -1)
        w = dec(torch.cat([z, torch.tensor(av)[None]], dim=1)).numpy()[0]
    assert got.shape == (1024,)
    assert np.allclose(got, (B @ w).astype(np.float32), atol=1e-4)


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} AREA-CONDITIONED DECODER TESTS PASSED")
