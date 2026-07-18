# All this code is from Claude
"""Contracts for the area common-mode JEPA. Synthetic data only.

Run: python -m src.tests.test_area_commonmode_jepa
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    area_to_chip,
    ensure_area_column,
    group_statistics,
    min_valid_stars,
    valid_k_values,
)
from src.instrument_v2.area_commonmode_jepa import (
    AreaCommonModeJEPA,
    commonmode_loss,
    load_group_jepa_warmstart,
)
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.group_level_jepa import GroupMeanInstrumentJEPA
from src.tests.test_sector14_jepa import synthetic_frame


def frame_with_areas(n_per_chip=12, n_cad=64, n_rings=2):
    """Synthetic frame + area column built from the EXISTING code convention
    (camera*100 + ccd*10 + ring), rings cycled within each chip."""
    frame = synthetic_frame(n_per_chip=n_per_chip, n_cad=n_cad)
    ring = (frame.groupby(["camera", "ccd"]).cumcount() % n_rings) + 1
    frame["area"] = frame["camera"] * 100 + frame["ccd"] * 10 + ring
    return frame


def make_dataset(grouping="area", k=4, frame=None):
    frame = frame if frame is not None else frame_with_areas()
    t0 = min(float(np.min(t)) for t in frame["time"])
    t1 = max(float(np.max(t)) for t in frame["time"])
    return Sector14GroupStatDataset(frame, set(frame["TIC"].astype(str)),
                                    (t0, t1), grouping, k, grid_length=64)


def tiny_model():
    torch.manual_seed(0)
    return AreaCommonModeJEPA(n_tokens=4, token_dim=4, d_model=8, n_layers=1)


# ------------------------------------------------------------------- areas
def test_existing_area_code_is_reused():
    frame = frame_with_areas()
    # column already present -> used verbatim, no re-derivation
    out = ensure_area_column(frame)
    assert (out["area"] == frame["area"]).all()
    # missing column -> merged from the existing *_area source on (TIC, sector)
    source = frame[["TIC", "sector", "area"]].copy()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "source_area.parquet")
        source.to_parquet(path)
        merged = ensure_area_column(frame.drop(columns=["area"]),
                                    area_source=path)
    assert (merged.set_index("TIC")["area"]
            == frame.set_index("TIC")["area"]).all()
    # code convention: 412 = camera 4, ccd 1, ring 2 -> chip_index(4, 1)
    assert area_to_chip(412) == chip_index(4, 1)


# ----------------------------------------------------------------- sampling
def test_context_and_targets_disjoint_same_group():
    for grouping in ("chip", "area"):
        dataset = make_dataset(grouping=grouping)
        np.random.seed(3)
        for _ in range(300):
            context, targets, group = dataset._sample_item()
            assert context not in set(targets.tolist())
            assert len(set(targets.tolist())) == dataset.k
            assert dataset.tics[context] not in set(dataset.tics[targets])
            assert dataset.group_labels[context] == group
            assert np.all(dataset.group_labels[targets] == group)


def test_deterministic_sampling_for_fixed_seed():
    dataset = make_dataset()
    np.random.seed(11)
    first = [dataset._sample_item() for _ in range(20)]
    np.random.seed(11)
    second = [dataset._sample_item() for _ in range(20)]
    for (c1, t1, g1), (c2, t2, g2) in zip(first, second):
        assert c1 == c2 and g1 == g2 and np.array_equal(t1, t2)


def test_valid_k_values_skips_starved_groupings():
    counts = {i: 9 for i in range(10)}          # 10 groups of 9 stars
    assert valid_k_values(counts, (8, 16, 32)) == [8]


# --------------------------------------------------------------- statistics
def test_median_mad_ignore_masked_values():
    k = 6
    flux = np.zeros((k, 8), dtype=np.float32)
    mask = np.ones_like(flux)
    flux[:, 0] = [1, 2, 3, 4, 5, 100]
    mask[5, 0] = 0.0                             # the 100 is UNOBSERVED
    flux[5, 0] = 100.0
    median, log_mad, valid, n_obs = group_statistics(flux, mask, min_valid_stars(k))
    assert n_obs[0] == 5
    assert median[0] == 3.0                      # median of 1..5, 100 excluded
    # zero-filled unobserved bins never contribute:
    mask[:, 1] = 0.0
    median2, _, valid2, _ = group_statistics(flux, mask, min_valid_stars(k))
    assert valid2[1] == 0.0 and median2[1] == 0.0


def test_minimum_coverage_enforced():
    k = 8
    threshold = min_valid_stars(k)               # max(4, 4) = 4
    flux = np.random.default_rng(0).normal(size=(k, 5)).astype(np.float32)
    mask = np.zeros_like(flux)
    mask[:threshold - 1, 0] = 1.0                # below threshold -> invalid
    mask[:threshold, 1] = 1.0                    # exactly threshold -> valid
    mask[:, 2] = 1.0                             # full coverage -> valid
    _, _, valid, _ = group_statistics(flux, mask, threshold)
    assert valid.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_statistics_permutation_invariant_over_target_stars():
    rng = np.random.default_rng(1)
    flux = rng.normal(size=(8, 32)).astype(np.float32)
    mask = (rng.random((8, 32)) > 0.3).astype(np.float32)
    perm = rng.permutation(8)
    a = group_statistics(flux, mask, 4)
    b = group_statistics(flux[perm], mask[perm], 4)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, rtol=1e-6)


# -------------------------------------------------------------------- model
def test_online_gets_gradients_ema_does_not():
    model = tiny_model()
    flux = torch.randn(3, 64)
    mask = torch.ones_like(flux)
    outputs = model(flux, mask, flux + 0.1, flux * 0.0, mask,
                    target="median_mad")
    loss, _ = commonmode_loss(*outputs[:4], outputs[4], mask)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.predictor_median.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_ema_update_moves_target_toward_online():
    model = tiny_model()
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target()
    moved = [(after - b).abs().sum() > 0 for b, after in
             zip(before, model.target_encoder.parameters())]
    assert any(moved)
    # blend direction: target moved toward online by (1 - momentum)
    online0 = next(iter(model.context_encoder.parameters()))
    target0 = next(iter(model.target_encoder.parameters()))
    expected = model.momentum * before[0] + (1 - model.momentum) * online0
    torch.testing.assert_close(target0, expected)


def test_individual_representation_is_downstream_interface():
    model = tiny_model().eval()
    flux = torch.randn(5, 64)
    tokens = model.encode(flux, torch.ones_like(flux), view="online")
    assert tokens.shape == (5, 4, 4)             # per-star tokens, no grouping
    outputs = model(flux, torch.ones_like(flux), flux, flux,
                    torch.ones_like(flux), target="median")
    assert outputs[4].shape == (5, 4, 4)         # context tokens are per-star
    assert outputs[2] is None and outputs[3] is None   # median arm: no MAD head


def test_warmstart_copies_online_encoder_into_both():
    torch.manual_seed(1)
    source = GroupMeanInstrumentJEPA(n_tokens=4, token_dim=4, d_model=8,
                                     n_layers=1, dropout=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "group.pth")
        torch.save(source.state_dict(), ckpt)
        selection = os.path.join(tmp, "selection.json")
        with open(selection, "w") as fh:
            json.dump({"checkpoint": ckpt}, fh)
        model = tiny_model()
        load_group_jepa_warmstart(model, selection)
    for ours, theirs in zip(model.context_encoder.parameters(),
                            source.context_encoder.parameters()):
        torch.testing.assert_close(ours, theirs)
    for ours, theirs in zip(model.target_encoder.parameters(),
                            source.context_encoder.parameters()):
        torch.testing.assert_close(ours, theirs)   # EMA starts as exact copy


# ---------------------------------------------------------- training smoke
def test_no_test_tic_enters_dataset():
    frame = frame_with_areas()
    test_tics = set(frame["TIC"].astype(str))
    allowed = {t for t in test_tics if not t.endswith(("0", "1"))}
    held_out = test_tics - allowed
    dataset = make_dataset(frame=frame[frame["TIC"].isin(allowed)])
    assert not set(dataset.tics) & held_out


def test_smoke_training_finite_both_targets():
    dataset = make_dataset(grouping="area", k=4)
    for target in ("median", "median_mad"):
        model = tiny_model()
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-3)
        np.random.seed(0)
        for _ in range(3):
            batch = [torch.stack(t) for t in
                     zip(*(dataset[i][:6] for i in range(8)))]
            ctx_f, ctx_m, median, log_mad, valid, _ = batch
            outputs = model(ctx_f, ctx_m, median, log_mad, valid, target=target)
            loss, parts = commonmode_loss(*outputs[:4], outputs[4], valid,
                                          mad_weight=0.25, var_weight=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()
            assert torch.isfinite(loss), f"non-finite loss in {target} arm"
            assert np.isfinite(parts["median_loss"])
        if target == "median_mad":
            assert np.isfinite(parts["mad_loss"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} AREA COMMON-MODE TESTS PASSED")
