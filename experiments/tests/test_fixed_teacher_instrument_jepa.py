# All this code is from Claude
"""Contracts for the fixed-regional-teacher pilot. Synthetic data only.

Run: python -m src.tests.test_fixed_teacher_instrument_jepa
"""

import numpy as np
import torch

from src.instrument_v2.area_commonmode_dataset import (
    group_statistics,
    min_valid_stars,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA,
    fixed_teacher_loss,
)
from src.instrument_v2.regional_group_teacher import (
    AreaGroupPairDataset,
    RegionalGroupTeacher,
    state_hash,
)
from src.loss_function.gapblind_fix import gapblind_loss
from src.tests.test_area_commonmode_jepa import frame_with_areas, make_dataset


def big_area_dataset(k=4):
    return make_dataset(grouping="area", k=k,
                        frame=frame_with_areas(n_per_chip=16))


def tiny_teacher():
    torch.manual_seed(0)
    return RegionalGroupTeacher(n_tokens=4, token_dim=4, d_model=8, n_layers=1)


def tiny_student():
    torch.manual_seed(1)
    return FixedTeacherInstrumentJEPA(n_tokens=4, token_dim=4, d_model=8,
                                      n_layers=1)


def teacher_batch(pairs, size=6):
    items = [pairs[i] for i in range(size)]
    return [torch.stack(t) for t in zip(*items)]


# ------------------------------------------------------------------ data
def test_teacher_groups_are_unique_same_area_stars():
    pairs = AreaGroupPairDataset(big_area_dataset())
    base = pairs.base
    np.random.seed(0)
    for _ in range(100):
        area = pairs.eligible[np.random.randint(len(pairs.eligible))]
        rows = np.random.choice(base.group_rows[area], size=2 * base.k,
                                replace=False)
        assert len(set(rows.tolist())) == 2 * base.k          # unique stars
        assert len(set(base.tics[rows])) == 2 * base.k        # unique TICs
        assert np.all(base.group_labels[rows] == area)        # one area


def test_context_star_excluded_from_student_targets():
    dataset = big_area_dataset()
    np.random.seed(1)
    for _ in range(200):
        context, targets, _ = dataset._sample_item()
        assert context not in set(targets.tolist())


def test_median_mad_ignore_masked_values_reference():
    flux = np.zeros((8, 4), dtype=np.float32)
    mask = np.ones_like(flux)
    flux[:, 0] = [1, 2, 3, 4, 5, 6, 7, 1000]
    mask[7, 0] = 0.0
    median, _, _, n_obs = group_statistics(flux, mask, min_valid_stars(8))
    assert n_obs[0] == 7 and median[0] == 4.0     # 1000 never contributes


def test_no_test_tics_enter():
    frame = frame_with_areas(n_per_chip=16)
    all_tics = set(frame["TIC"].astype(str))
    allowed = {t for t in all_tics if not t.endswith(("4", "5"))}
    dataset = make_dataset(grouping="area", k=4,
                           frame=frame[frame["TIC"].isin(allowed)])
    assert not set(dataset.tics) & (all_tics - allowed)


# ----------------------------------------------------------------- model
def test_student_and_teacher_token_shapes_match():
    model = tiny_student().eval()
    star = torch.randn(3, 64)
    stats = torch.randn(3, 64, 2)
    valid = torch.ones(3, 64)
    with torch.no_grad():
        prediction, target, tokens = model(star, torch.ones_like(star),
                                           stats, valid)
    assert prediction.shape == target.shape == tokens.shape == (3, 4, 4)


def test_gradients_reach_student_and_predictor_not_teacher():
    model = tiny_student()
    star = torch.randn(4, 64)
    stats = torch.randn(4, 64, 2)
    valid = torch.ones(4, 64)
    prediction, target, tokens = model(star, torch.ones_like(star), stats, valid)
    loss = fixed_teacher_loss(prediction, target, tokens, valid)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())


def test_teacher_hash_unchanged_after_optimizer_steps():
    model = tiny_student()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    before_teacher = model.teacher_hash()
    before_student = state_hash(model.student)
    for _ in range(3):
        star = torch.randn(4, 64)
        stats = torch.randn(4, 64, 2)
        valid = torch.ones(4, 64)
        prediction, target, tokens = model(star, torch.ones_like(star),
                                           stats, valid)
        loss = fixed_teacher_loss(prediction, target, tokens, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert model.teacher_hash() == before_teacher      # frozen: bit-identical
    assert state_hash(model.student) != before_student  # student DID train


def test_teacher_stays_in_eval_mode():
    model = tiny_student()
    model.train()
    assert model.student.training and not model.teacher.training


# ----------------------------------------------------------------- smoke
def test_teacher_smoke_epoch_finite():
    pairs = AreaGroupPairDataset(big_area_dataset())
    model = tiny_teacher()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3)
    np.random.seed(0)
    for _ in range(3):
        stats_a, valid_a, stats_b, valid_b, _, _ = teacher_batch(pairs)
        prediction, target, context_tokens = model(stats_a, valid_a,
                                                   stats_b, valid_b)
        loss = gapblind_loss(prediction, target, context_tokens,
                             target_mask=valid_b, var_weight=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()
        assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.online_encoder.parameters())
    assert all(p.grad is None for p in model.ema_encoder.parameters())


def test_student_smoke_epoch_finite():
    dataset = big_area_dataset()
    model = tiny_student()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    np.random.seed(0)
    for _ in range(3):
        items = [dataset[i][:5] for i in range(6)]
        ctx_f, ctx_m, median, log_mad, valid = \
            [torch.stack(t) for t in zip(*items)]
        stats = torch.stack([median, log_mad], dim=-1)
        prediction, target, tokens = model(ctx_f, ctx_m, stats, valid)
        loss = fixed_teacher_loss(prediction, target, tokens, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


def test_finetune_smoke_step_finite():
    from src.instrument_v2.train_area_commonmode_finetune import Classifier
    torch.manual_seed(2)
    model = tiny_student()
    head = torch.nn.Linear(4 * 4, 16)
    classifier = Classifier(model.student, head)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    flux = torch.randn(8, 64)
    mask = torch.ones_like(flux)
    labels = torch.randint(0, 16, (8,))
    logits = classifier(flux, mask)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


# ----------------------------------------------- transformer predictor
def tiny_tx_student():
    torch.manual_seed(3)
    return FixedTeacherInstrumentJEPA(n_tokens=4, token_dim=4, d_model=8,
                                      n_layers=1,
                                      predictor_type="transformer")


def test_transformer_predictor_shapes():
    from src.instrument_v2.fixed_teacher_instrument_jepa import (
        TransformerPredictor,
    )
    predictor = TransformerPredictor()          # spec dims: 16 tokens x 16
    tokens = torch.randn(2, 16, 16)
    assert predictor(tokens).shape == (2, 16, 16)
    model = tiny_tx_student().eval()
    star = torch.randn(3, 64)
    with torch.no_grad():
        prediction, target, tokens = model(star, torch.ones_like(star),
                                           torch.randn(3, 64, 2),
                                           torch.ones(3, 64))
    assert prediction.shape == target.shape == tokens.shape == (3, 4, 4)


def test_transformer_gradients_student_and_predictor_not_teacher():
    model = tiny_tx_student()
    star = torch.randn(4, 64)
    outputs = model(star, torch.ones_like(star), torch.randn(4, 64, 2),
                    torch.ones(4, 64))
    loss = fixed_teacher_loss(*outputs, torch.ones(4, 64))
    before = model.teacher_hash()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.teacher_hash() == before


def test_transformer_masked_batches_stay_finite():
    model = tiny_tx_student().eval()
    star = torch.randn(3, 64)
    star_mask = torch.ones_like(star)
    star_mask[:, 32:48] = 0.0                    # one fully-invalid token
    with torch.no_grad():
        prediction, target, _ = model(star, star_mask,
                                      torch.randn(3, 64, 2), star_mask)
    assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
    padding = model._token_padding(star_mask)
    assert padding.shape == (3, 4) and padding[:, 2].all()
    assert not padding[:, 0].any()


def test_frozen_probe_updates_only_classifier():
    import numpy as np
    from src.instrument_v2.train_group_level_jepa import fast_probe
    model = tiny_tx_student().eval()
    before = state_hash(model)
    rng = np.random.default_rng(0)
    train_z = rng.normal(size=(64, 16))
    val_z = rng.normal(size=(32, 16))
    labels = rng.integers(0, 4, 64)
    fast_probe(train_z, labels, val_z, rng.integers(0, 4, 32))
    assert state_hash(model) == before           # probe touched nothing


# --------------------------------------- encoder-benchmark protocol
def test_selection_metric_uses_encoder_view():
    from src.instrument_v2.train_fixed_teacher_instrument_jepa import (
        selection_metric,
    )
    # online view: encoder camCCD decides, transformer output ignored
    assert selection_metric("online", 0.30, 0.99) == 0.30
    assert selection_metric("predicted", 0.30, 0.99) == 0.99


def test_pass_verdict_uses_encoder_not_transformer():
    from src.instrument_v2.eval_transformer_predictor_screen import (
        compute_verdict,
    )
    def results(encoder, random, tx_out):
        return {"tx_jepa_encoder": {"val_camccd_bacc": encoder},
                "random_s4d": {"val_camccd_bacc": random},
                "mlp_jepa_encoder": {"val_camccd_bacc": 0.43},
                "tx_jepa_transformer": {"val_camccd_bacc": tx_out},
                "random_s4d_tx": {"val_camccd_bacc": 0.40}}
    # huge transformer output cannot rescue a losing encoder
    _, verdict = compute_verdict(results(0.40, 0.42, 0.99))
    assert verdict == "FAIL"
    # winning encoder passes even with a terrible transformer output
    diffs, verdict = compute_verdict(results(0.45, 0.42, 0.01))
    assert verdict == "PASS"
    assert abs(diffs["encoder_minus_random_s4d"] - 0.03) < 1e-9


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} FIXED-TEACHER TESTS PASSED")
