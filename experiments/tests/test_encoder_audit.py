# All this code is from Claude
"""Safeguard tests for the online-vs-EMA encoder audit.

Synthetic data only. Run: python -m src.tests.test_encoder_audit
"""

import json
import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.ablation_config import (
    BACKBONE_LRS, ONLINE_FT_ARMS, SEEDS, map_online_finetune_task,
)
from src.instrument_v2.encoder_source import (
    assert_same_architecture, encode_features, encoders_identical,
    extract_encoder, param_distance,
)
from src.instrument_v2.train_sector14_matched_finetune import make_head
from src.loss_function.gapblind_fix import build_gapblind_jepa
from src.worked_folder.instrument.instrument_jepa import InstrumentJEPA


def test_extract_returns_correct_modules():
    torch.manual_seed(0)
    model = build_gapblind_jepa()
    assert extract_encoder(model, "online") is model.context_encoder
    assert extract_encoder(model, "ema") is model.target_encoder
    try:
        extract_encoder(model, "target")
        assert False, "bad source accepted"
    except ValueError:
        pass


def test_random_init_identical_and_diverges_after_training():
    torch.manual_seed(1)
    model = build_gapblind_jepa()
    assert_same_architecture(model)
    assert encoders_identical(model), "fresh model: online != ema"
    assert param_distance(model) == 0.0
    # one online gradient step + EMA update -> encoders must differ
    x, m = torch.randn(4, 1024), torch.ones(4, 1024)
    loss = model.context_encoder(x.unsqueeze(-1), m).sum()
    opt = torch.optim.SGD(model.context_encoder.parameters(), lr=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
    model.update_target()
    assert not encoders_identical(model)
    assert param_distance(model) > 0


def test_online_evaluation_never_calls_model_encode():
    torch.manual_seed(2)
    model = build_gapblind_jepa()

    def boom(*a, **k):
        raise AssertionError("model.encode() called during online evaluation")
    model.encode = boom                          # instance-level trap
    X = np.random.randn(8, 1024).astype(np.float32)
    M = np.ones_like(X)
    Z = encode_features(model, "online", X, M, torch.device("cpu"))
    assert Z.shape[0] == 8
    Z2 = encode_features(model, "ema", X, M, torch.device("cpu"))
    assert Z2.shape[0] == 8                      # ema path also bypasses encode()


def test_both_sources_identical_preprocessing():
    """Same X/M + equal weights (fresh model) -> bit-identical features from
    both sources: proves the preprocessing/call convention is shared."""
    torch.manual_seed(3)
    model = build_gapblind_jepa()
    model.eval()
    X = np.random.randn(6, 1024).astype(np.float32)
    M = (np.random.rand(6, 1024) > 0.1).astype(np.float32)
    Z_online = encode_features(model, "online", X, M, torch.device("cpu"))
    Z_ema = encode_features(model, "ema", X, M, torch.device("cpu"))
    np.testing.assert_array_equal(Z_online, Z_ema)


def test_online_checkpoint_selection_validation_only():
    from src.instrument_v2.select_pretrain_checkpoints import load_trainval
    from src.tests.test_sector14_jepa import synthetic_frame
    df = synthetic_frame()
    all_tics = sorted(set(df["TIC"].astype(str)))
    train, val, test = set(all_tics[:60]), set(all_tics[60:80]), set(all_tics[80:])
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    X, M, chips, is_train = load_trainval(df, train, val, test, (t0, t1))
    assert len(X) == len(train | val)            # selector data path excludes test
    try:
        load_trainval(df, train | set(list(test)[:1]), val, test, (t0, t1))
        assert False, "test TIC accepted by selection data path"
    except RuntimeError:
        pass


def test_finetune_loads_online_encoder():
    from src.instrument_v2.train_online_finetune import make_online_encoder
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(4)
        model = build_gapblind_jepa()
        # perturb ONLINE only so the two encoders are distinguishable
        with torch.no_grad():
            next(model.context_encoder.parameters()).add_(1.0)
        ckpt = os.path.join(tmp, "m.pth")
        torch.save(model.state_dict(), ckpt)
        manifest = os.path.join(tmp, "sel.json")
        with open(manifest, "w") as fh:
            json.dump({"arms": {"supcon": {"0": {"checkpoint": ckpt}}}}, fh)
        enc = make_online_encoder("supcon", 0, manifest_path=manifest)
        got = next(enc.parameters()).detach()
        want_online = next(model.context_encoder.parameters()).detach()
        want_ema = next(model.target_encoder.parameters()).detach()
        assert torch.equal(got, want_online), "did not load the online encoder"
        assert not torch.equal(got, want_ema), "loaded the EMA encoder instead"


def test_head_init_identical_to_abl1():
    h_audit = make_head(16, seed=1, target="camccd")
    torch.manual_seed(31337)                     # pollute RNG
    h_abl1 = make_head(16, seed=1, target="camccd")
    assert torch.equal(h_audit.weight, h_abl1.weight)
    assert torch.equal(h_audit.bias, h_abl1.bias)


def test_exactly_27_online_finetune_tasks():
    tasks = [map_online_finetune_task(i) for i in range(27)]
    combos = {(t["INIT_ARM"], t["SEED"], t["BACKBONE_LR"]) for t in tasks}
    assert len(combos) == 27
    assert combos == {(a, s, lr) for a in ONLINE_FT_ARMS for s in SEEDS
                      for lr in BACKBONE_LRS}
    for bad in (-1, 27):
        try:
            map_online_finetune_task(bad); assert False
        except IndexError:
            pass


def test_final_audit_requires_gate():
    from src.instrument_v2.eval_encoder_source_final import require_final_gate
    os.environ.pop("FINAL_EVAL", None)
    try:
        require_final_gate()
        assert False, "final audit ran without FINAL_EVAL=YES"
    except RuntimeError:
        pass
    os.environ["FINAL_EVAL"] = "YES"
    require_final_gate()
    os.environ.pop("FINAL_EVAL", None)


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
