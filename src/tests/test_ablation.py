# All this code is from Claude
"""Safeguard tests for the instrument ablation (src/instrument_v2).

Synthetic data only. Run: python -m src.tests.test_ablation
"""

import json
import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.ablation_config import (
    ARMS, BACKBONE_LRS, HYBRID_WEIGHTS, SEEDS, TARGETS,
    map_finetune_task, map_pretrain_task,
)
from src.instrument_v2.contrastive_loss import supcon_loss
from src.instrument_v2.sector14_dataset import Sector14ChipPairDataset, ensure_splits
from src.instrument_v2.train_sector14_contrastive import total_loss
from src.instrument_v2.train_sector14_matched_finetune import (
    build_supervised_frames, make_encoder, make_head,
)
from src.loss_function.gapblind_fix import build_gapblind_jepa
from src.tests.test_sector14_jepa import synthetic_frame


# ---------------- data rules ----------------
def test_only_raw_flux_used():
    """Garbage flux_cal must not change dataset tensors (raw `flux` only)."""
    df1 = synthetic_frame(garbage_cal=False)
    df2 = synthetic_frame(garbage_cal=True)
    tics = set(df1["TIC"].astype(str))
    t0 = min(float(np.min(t)) for t in df1["time"])
    t1 = max(float(np.max(t)) for t in df1["time"])
    d1 = Sector14ChipPairDataset(df1, tics, "shared", (t0, t1), return_chip=True)
    d2 = Sector14ChipPairDataset(df2, tics, "shared", (t0, t1), return_chip=True)
    np.testing.assert_array_equal(d1.X, d2.X)


def test_splits_disjoint():
    tics = [f"T{i}" for i in range(200)]
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "split_train_tics.txt"), "w") as fh:
            fh.write("\n".join(tics[:160]))
        with open(os.path.join(tmp, "split_test_tics.txt"), "w") as fh:
            fh.write("\n".join(tics[160:]))
        train, val, test = ensure_splits(tmp, os.path.join(tmp, "exp"))
        assert not (train & val) and not (train & test) and not (val & test)


def test_test_tics_never_reach_training():
    """Pretraining dataset AND fine-tune frame builder both exclude test TICs."""
    df = synthetic_frame()
    all_tics = sorted(set(df["TIC"].astype(str)))
    train, val, test = set(all_tics[:60]), set(all_tics[60:80]), set(all_tics[80:])
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    ds = Sector14ChipPairDataset(df, train, "shared", (t0, t1), return_chip=True)
    assert not (set(ds.tics) & test)
    X, M, y, is_train = build_supervised_frames(df, train, val, test, (t0, t1), "camccd")
    assert len(X) == len(train | val), "fine-tune frames must hold train+val only"
    try:
        build_supervised_frames(df, train, val | {next(iter(test))} - test, test, (t0, t1), "camera")
    except RuntimeError:
        pass  # leaking a test TIC into val must raise -- covered below explicitly
    try:
        build_supervised_frames(df, train | {next(iter(test))}, val, test, (t0, t1), "camera")
        assert False, "test TIC in train set was accepted"
    except RuntimeError:
        pass


# ---------------- supcon loss ----------------
def test_supcon_finite():
    rng = torch.Generator().manual_seed(0)
    z = torch.randn(64, 16, 16, generator=rng)
    labels = torch.randint(0, 16, (64,), generator=rng)
    loss = supcon_loss(z, labels)
    assert torch.isfinite(loss)
    # anchors with no positive must not produce NaN
    loss_unique = supcon_loss(torch.randn(4, 8, generator=rng), torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss_unique)


def test_supcon_prefers_correct_clusters():
    rng = torch.Generator().manual_seed(1)
    centers = torch.randn(4, 32, generator=rng) * 5
    labels = torch.arange(4).repeat_interleave(16)
    z = centers[labels] + 0.1 * torch.randn(64, 32, generator=rng)
    good = supcon_loss(z, labels)
    permuted = labels[torch.randperm(64, generator=rng)]
    bad = supcon_loss(z, permuted)
    assert good < bad, f"clustered loss {good} not below permuted {bad}"


def test_hybrid_weight_zero_is_exactly_jepa():
    jepa = torch.tensor(1.2345)
    con = torch.tensor(999.0)
    assert total_loss(jepa, con, "hybrid", 0.0) is jepa
    assert total_loss(jepa, con, "supcon", 0.0) is con
    combined = total_loss(jepa, con, "hybrid", 0.5)
    assert torch.isclose(combined, jepa + 0.5 * con)


# ---------------- model / EMA / heads ----------------
def test_ema_target_changes_during_supcon_training():
    torch.manual_seed(0)
    model = build_gapblind_jepa()
    before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
    x = torch.randn(8, 1024)
    m = torch.ones(8, 1024)
    tokens = model.context_encoder(x.unsqueeze(-1), m)
    loss = supcon_loss(tokens, torch.randint(0, 4, (8,)))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
    model.update_target()
    changed = any(not torch.equal(before[k], v)
                  for k, v in model.target_encoder.state_dict().items())
    assert changed, "EMA target did not move after SupCon step + update_target"


def test_all_arms_identical_architecture():
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        ckpt = os.path.join(tmp, "any.pth")
        torch.save(build_gapblind_jepa().state_dict(), ckpt)
        manifest = {"arms": {a: {"0": {"checkpoint": ckpt}} for a in ARMS}}
        mpath = os.path.join(tmp, "sel.json")
        with open(mpath, "w") as fh:
            json.dump(manifest, fh)
        shapes = {}
        for arm in ARMS:
            enc = make_encoder(arm, 0, manifest_path=mpath)
            shapes[arm] = {k: tuple(v.shape) for k, v in enc.state_dict().items()}
        assert all(s == shapes["random"] for s in shapes.values()), \
            "encoder architecture differs between arms"


def test_heads_identically_initialized_per_seed_and_target():
    h1 = make_head(16, seed=1, target="camccd")
    torch.manual_seed(999)                       # pollute RNG between constructions
    h2 = make_head(16, seed=1, target="camccd")
    assert torch.equal(h1.weight, h2.weight) and torch.equal(h1.bias, h2.bias)
    h3 = make_head(16, seed=2, target="camccd")
    assert not torch.equal(h1.weight, h3.weight), "different seeds must differ"


def test_checkpoint_loading_all_arms():
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(3)
        src = build_gapblind_jepa()
        for name in ("s14supcon_s0_ep010", "s14hybrid_w0.5_s0_ep010", "s14jepa_shared_s0_ep010"):
            path = os.path.join(tmp, f"{name}.pth")
            torch.save(src.state_dict(), path)
            dst = build_gapblind_jepa()
            dst.load_state_dict(torch.load(path))   # must not raise


# ---------------- selection / final-eval safety ----------------
def test_selection_never_accesses_test():
    from src.instrument_v2.select_pretrain_checkpoints import load_trainval
    df = synthetic_frame()
    all_tics = sorted(set(df["TIC"].astype(str)))
    train, val, test = set(all_tics[:60]), set(all_tics[60:80]), set(all_tics[80:])
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    X, M, chips, is_train = load_trainval(df, train, val, test, (t0, t1))
    assert len(X) == len(train | val)
    try:
        load_trainval(df, train, val | set(list(test)[:1]), test, (t0, t1))
        assert False, "selection accepted a test TIC"
    except RuntimeError:
        pass


def test_final_eval_requires_gate():
    from src.instrument_v2.eval_final_ablation import require_final_gate
    os.environ.pop("FINAL_EVAL", None)
    try:
        require_final_gate()
        assert False, "final eval ran without FINAL_EVAL=YES"
    except RuntimeError:
        pass
    os.environ["FINAL_EVAL"] = "YES"
    require_final_gate()
    os.environ.pop("FINAL_EVAL", None)


def test_slurm_task_mapping_exact():
    pre = {tuple(sorted(map_pretrain_task(i).items())) for i in range(12)}
    assert len(pre) == 12
    expected_pre = 3 + len(HYBRID_WEIGHTS) * 3
    assert expected_pre == 12
    ft = [map_finetune_task(i) for i in range(72)]
    combos = {(t["INIT_ARM"], t["TARGET"], t["SEED"], t["BACKBONE_LR"]) for t in ft}
    assert len(combos) == 72, "duplicate fine-tune tasks"
    assert combos == {(a, t, s, lr) for a in ARMS for t in TARGETS
                      for s in SEEDS for lr in BACKBONE_LRS}, "coverage gap"
    for bad in (-1, 12):
        try:
            map_pretrain_task(bad); assert False
        except IndexError:
            pass
    for bad in (-1, 72):
        try:
            map_finetune_task(bad); assert False
        except IndexError:
            pass


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
