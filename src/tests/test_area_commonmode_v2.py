# All this code is from Claude
"""Contracts for area_commonmode_v2 (raw diagnostic + decorrelation).

Run: python -m src.tests.test_area_commonmode_v2
"""

import json
import os
import tempfile

import numpy as np
import torch

import src.instrument_v2.report_area_commonmode_v2 as report_v2
from src.instrument_v2.area_commonmode_jepa import (
    commonmode_loss,
    covariance_penalty,
)
from src.instrument_v2.diagnose_raw_area_commonmode import (
    ShuffledAreaView,
    masked_correlation,
    pair_similarities,
)
from src.tests.test_area_commonmode_jepa import (
    frame_with_areas,
    make_dataset,
    tiny_model,
)


# ------------------------------------------------------- raw diagnostic
def test_similarity_uses_only_mutually_observed_cadences():
    a = np.array([1.0, 2.0, 3.0, 99.0, -99.0])
    b = np.array([1.0, 2.0, 3.0, -55.0, 42.0])
    mutual = np.array([1, 1, 1, 0, 0], dtype=bool)
    # identical on mutual cadences, wildly different outside -> r must be 1
    padded_a = np.concatenate([a, np.linspace(0, 1, 8)])
    padded_b = np.concatenate([b, np.linspace(0, 1, 8)])
    padded_mutual = np.concatenate([mutual, np.ones(8, dtype=bool)])
    assert masked_correlation(padded_a, padded_b, padded_mutual) > 0.999
    # under 8 mutual cadences -> nan, never a fabricated similarity
    assert np.isnan(masked_correlation(a, b, mutual))


def big_area_dataset(k=4):
    # disjoint two-group sampling needs >= 2K stars per area:
    # 16 stars/chip over 2 rings -> 8 per area = exactly 2K at K=4.
    return make_dataset(grouping="area", k=k,
                        frame=frame_with_areas(n_per_chip=16))


def test_pair_similarities_on_dataset_groups():
    dataset = big_area_dataset()
    np.random.seed(0)
    rows_a, rows_b, _ = dataset.sample_disjoint_same_group()
    med_r, mad_r, combined, coverage = pair_similarities(dataset, rows_a, rows_b)
    assert np.isfinite(combined) and 0.0 <= coverage <= 1.0


def test_same_and_cross_area_sampling():
    dataset = big_area_dataset()
    np.random.seed(1)
    for _ in range(100):
        rows_a, rows_b, group = dataset.sample_disjoint_same_group()
        assert not set(rows_a) & set(rows_b)
        assert np.all(dataset.group_labels[rows_a] == group)
        assert np.all(dataset.group_labels[rows_b] == group)
        draw = dataset.sample_cross_group()
        rows_a, rows_b, (g1, g2) = draw
        assert g1 != g2
        # same parent chip = same camera x CCD, different area
        assert (g1 // 10) == (g2 // 10) or (g1 // 100 == g2 // 100)
        assert np.all(dataset.group_labels[rows_a] == g1)
        assert np.all(dataset.group_labels[rows_b] == g2)


def test_shuffled_area_control_mixes_true_areas():
    dataset = big_area_dataset()
    shuffled = ShuffledAreaView(dataset, seed=0)
    np.random.seed(2)
    mixed = 0
    for _ in range(50):
        rows_a, rows_b, _ = shuffled.sample_disjoint_same_group()
        true_areas = set(dataset.group_labels[np.concatenate([rows_a, rows_b])])
        if len(true_areas) > 1:
            mixed += 1
    assert mixed > 25, "shuffled control fails to break area structure"


# --------------------------------------------------- covariance penalty
def test_covariance_zero_for_decorrelated_features():
    torch.manual_seed(0)
    n, d = 512, 8
    raw = torch.randn(n, d).double()
    raw = raw - raw.mean(dim=0)
    u, s, _ = torch.linalg.svd(raw, full_matrices=False)
    z = (u * s).float()                          # PCA scores: exactly uncorrelated
    penalty = covariance_penalty(z.reshape(n, 2, d // 2))
    assert penalty.item() < 1e-6


def test_covariance_grows_for_duplicated_dimensions():
    torch.manual_seed(0)
    base = torch.randn(256, 4)
    independent = torch.randn(256, 8)
    duplicated = torch.cat([base, base], dim=1)  # perfectly correlated copies
    p_ind = covariance_penalty(independent.reshape(256, 2, 4))
    p_dup = covariance_penalty(duplicated.reshape(256, 2, 4))
    assert p_dup.item() > 10 * p_ind.item()


def test_covariance_gradients_reach_online_encoder_only():
    model = tiny_model()
    flux = torch.randn(6, 64)
    mask = torch.ones_like(flux)
    outputs = model(flux, mask, flux, flux * 0.0, mask, target="median_mad")
    loss, parts = commonmode_loss(*outputs[:4], outputs[4], mask,
                                  mad_weight=0.0, var_weight=0.0,
                                  cov_weight=1.0)
    # only the covariance path plus median loss; cov must reach the encoder
    loss.backward()
    assert parts["cov_loss"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_cov_weight_zero_reproduces_v1_loss():
    model = tiny_model().eval()
    flux = torch.randn(6, 64)
    mask = torch.ones_like(flux)
    with torch.no_grad():
        outputs = model(flux, mask, flux, flux * 0.5, mask, target="median_mad")
        old, _ = commonmode_loss(*outputs[:4], outputs[4], mask,
                                 mad_weight=0.25, var_weight=0.5)
        new, parts = commonmode_loss(*outputs[:4], outputs[4], mask,
                                     mad_weight=0.25, var_weight=0.5,
                                     cov_weight=0.0)
    torch.testing.assert_close(old, new)
    assert parts["cov_loss"] >= 0.0            # recorded even when unweighted


# ------------------------------------------------------------ protocol
def test_no_test_tic_enters_v2_dataset():
    frame = frame_with_areas()
    all_tics = set(frame["TIC"].astype(str))
    allowed = {t for t in all_tics if not t.endswith(("2", "3"))}
    dataset = make_dataset(frame=frame[frame["TIC"].isin(allowed)])
    assert not set(dataset.tics) & (all_tics - allowed)


def test_failed_gates_still_produce_final_report():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "raw_region_diagnostic.json"), "w") as fh:
            json.dump({"passes": False,
                       "splits": {"val": {"same_minus_cross": {"combined_r":
                           {"mean": -0.01, "ci95": [-0.03, 0.01]}}}}}, fh)
        with open(os.path.join(tmp, "screen_selection.json"), "w") as fh:
            json.dump({"selected": None, "screen_table": [],
                       "refusal_reason": "raw region diagnostic FAILED"}, fh)
        original = report_v2.ART_DIR
        try:
            report_v2.ART_DIR = tmp
            report_v2.final_report()
        finally:
            report_v2.ART_DIR = original
        assert os.path.exists(os.path.join(tmp, "final_summary.md"))
        assert os.path.exists(os.path.join(tmp, "final_summary.json"))
        with open(os.path.join(tmp, "final_summary.json")) as fh:
            summary = json.load(fh)
        assert summary["screen"]["refusal_reason"]
        assert "test" in summary["test_untouched"].lower()


def test_smoke_training_finite_with_covariance():
    dataset = make_dataset(grouping="area", k=4)
    model = tiny_model()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3)
    np.random.seed(0)
    for _ in range(3):
        batch = [torch.stack(t) for t in
                 zip(*(dataset[i][:6] for i in range(8)))]
        ctx_f, ctx_m, median, log_mad, valid, _ = batch
        outputs = model(ctx_f, ctx_m, median, log_mad, valid,
                        target="median_mad")
        loss, parts = commonmode_loss(*outputs[:4], outputs[4], valid,
                                      mad_weight=0.25, var_weight=0.5,
                                      cov_weight=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()
        assert torch.isfinite(loss)
        assert np.isfinite(parts["cov_loss"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} AREA COMMON-MODE V2 TESTS PASSED")
