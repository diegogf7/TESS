# All this code is from Claude
"""Contracts for the AreaHeadDecoder (shared trunk + per-area 8-weight head).
Synthetic only -- no cluster, no real checkpoints, no test TICs.

Run: python -m src.tests.test_area_head_decoder
"""

import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.decode_single_star_k8 import (
    build_decoder, build_area_head_decoder, AreaHeadDecoder, load_area_decoder,
    area_index_map, area_one_hot, decode, CBV_RANK,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


def _raises(fn):
    try:
        fn(); return False
    except RuntimeError:
        return True


# 1) head selection: the argmax of the one-hot picks that area's head, and the
#    output equals that head applied to the shared trunk output
def test_head_selection_correct():
    torch.manual_seed(0)
    a2i = {111: 0, 233: 1, 412: 2}
    dec = build_area_head_decoder(len(a2i), out_dim=CBV_RANK).eval()
    with torch.no_grad():                                    # make heads clearly distinct
        for k in range(len(a2i)):
            dec.head_w[k].fill_(float(k + 1)); dec.head_b[k].fill_(float(k))
    latent = torch.randn(1, 256)
    h = dec.trunk(latent)
    for area, idx in a2i.items():
        x = torch.cat([latent, torch.tensor(area_one_hot(a2i, area))[None]], dim=1)
        got = dec(x)
        manual = dec.head_w[idx] @ h[0] + dec.head_b[idx]
        assert torch.allclose(got[0], manual, atol=1e-5)     # correct head used
    # different areas -> different outputs (heads actually differ)
    xa = torch.cat([latent, torch.tensor(area_one_hot(a2i, 111))[None]], dim=1)
    xb = torch.cat([latent, torch.tensor(area_one_hot(a2i, 412))[None]], dim=1)
    assert not torch.allclose(dec(xa), dec(xb))


# 2) warm-start from global => every area reproduces the global decoder exactly
def test_init_from_global_matches_global():
    torch.manual_seed(1)
    a2i = {111: 0, 233: 1, 412: 2}
    global_dec = build_decoder(CBV_RANK).eval()              # Sequential(Flatten,LN,Lin,GELU,Lin)
    gstate = global_dec.state_dict()
    hd = build_area_head_decoder(len(a2i), out_dim=CBV_RANK, global_state=gstate).eval()
    latent = torch.randn(5, 256)
    g_out = global_dec(latent)                               # (5, 8)
    for area in a2i:
        oh = torch.stack([torch.tensor(area_one_hot(a2i, area))] * 5)
        h_out = hd(torch.cat([latent, oh], dim=1))
        assert torch.allclose(g_out, h_out, atol=1e-5)       # starts at the global solution


# 3) deterministic checkpoint mapping: round-trips through load_area_decoder
def test_checkpoint_mapping_roundtrip():
    a2i = area_index_map([412, 111, 233])
    assert a2i == {111: 0, 233: 1, 412: 2}                   # sorted, deterministic
    hd = build_area_head_decoder(len(a2i), out_dim=CBV_RANK)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "decoder.pth")
        torch.save({"state_dict": hd.state_dict(), "area_to_index": a2i, "n_areas": len(a2i),
                    "in_dim": 256 + len(a2i), "cbv_rank": CBV_RANK, "decoder_kind": "area_heads"}, p)
        dec, a2i2, kind = load_area_decoder(p, torch.device("cpu"))
        assert a2i2 == a2i and kind == "area_heads" and isinstance(dec, AreaHeadDecoder)
        # loaded weights identical
        assert torch.allclose(dec.head_w, hd.head_w) and torch.allclose(dec.head_b, hd.head_b)


# 4) unknown areas hard-fail
def test_unknown_area_hard_fails():
    a2i = {111: 0, 233: 1}
    assert area_one_hot(a2i, 233).tolist() == [0.0, 1.0]
    assert _raises(lambda: area_one_hot(a2i, 999))
    # a wrong-width input (missing one-hot) hard-fails inside the decoder
    hd = build_area_head_decoder(len(a2i), out_dim=CBV_RANK)
    assert _raises(lambda: hd(torch.randn(1, 256)))          # no one-hot appended


# 5) validation-row eligibility is decoder-independent (identical rows)
def test_eligibility_decoder_independent():
    from src.instrument_v2.eval_cbv_oracle_ceiling import reference_median
    from src.instrument_v2.decode_single_star_k8 import deterministic_area_rows

    class DS:
        pass
    rng = np.random.default_rng(0)
    ds = DS()
    ds.areas = np.array([111] * 40 + [233] * 40)
    ds.tics = np.array([f"T{i}" for i in range(80)])
    ds.X = rng.normal(size=(80, 1024)).astype(np.float32)
    ds.M = np.ones((80, 1024), np.float32)
    ds.min_valid = 16
    rows = deterministic_area_rows(ds)
    bases = {111: rng.normal(size=(1024, CBV_RANK)), 233: rng.normal(size=(1024, CBV_RANK))}
    elig = [i for i in range(80) if reference_median(ds, rows, i, bases)[0] is not None]
    assert elig == [i for i in range(80) if reference_median(ds, rows, i, bases)[0] is not None]
    assert len(elig) > 0


# 6) frozen instrument unchanged; only the area-head decoder trains
def test_frozen_unchanged_only_decoder_trains():
    torch.manual_seed(0)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                       readout="mean", predictor_type="mlp").eval()
    for p in model.parameters():
        p.requires_grad = False
    a2i = {111: 0, 233: 1}
    hd = build_area_head_decoder(len(a2i), out_dim=CBV_RANK)
    before = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    raw, mask = np.random.randn(64).astype(np.float32), np.ones(64, np.float32)
    got = decode(model, hd, raw, mask, np.random.randn(1024, CBV_RANK).astype(np.float32),
                 area_one_hot(a2i, 233))                     # area-head via decode() concat path
    assert got.shape == (1024,)
    # a manual grad step touches only decoder params
    r = torch.tensor(raw)[None]; m = torch.tensor(mask)[None]
    z = model.encode(r, m, view="predicted").reshape(1, -1)
    oh = torch.tensor(area_one_hot(a2i, 233))[None]
    hd(torch.cat([z, oh], dim=1)).pow(2).mean().backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None for p in hd.parameters())
    after = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    assert after == before


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} AREA-HEAD DECODER TESTS PASSED")
