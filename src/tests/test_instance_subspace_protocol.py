# All this code is from Claude
"""Protocol-level tests for the Instance-to-Subspace experiment.

Run: python -m src.tests.test_instance_subspace_protocol
"""

import numpy as np
import torch

from src.instrument_v2.instance_subspace_jepa import build_instance_subspace
from src.tests.test_sector14_jepa import synthetic_frame


def test_frozen_probing_uses_encode_instrument():
    """The trainer's latent extraction must call encode_instrument()."""
    from src.instrument_v2 import train_instance_subspace_jepa as trainer
    model = build_instance_subspace("instance_mean")
    calls = []
    original = model.encode_instrument

    def spy(flux, observed_mask=None, source="online"):
        calls.append(source)
        return original(flux, observed_mask, source)
    model.encode_instrument = spy

    class TinyDS:
        X = np.random.randn(8, 1024).astype(np.float32)
        M = np.ones((8, 1024), dtype=np.float32)
    trainer.DEVICE = torch.device("cpu")
    Z = trainer.instrument_latents(model, TinyDS())
    assert calls == ["online"], "probing did not go through encode_instrument()"
    assert Z.shape == (8, 256)


def test_projector_is_part_of_representation():
    """encode_instrument must include the projector: zeroing the projector's
    final layer must change the representation."""
    torch.manual_seed(0)
    model = build_instance_subspace("instance_cov")
    x = torch.randn(4, 1024)
    m = torch.ones(4, 1024)
    z1 = model.encode_instrument(x, m)
    with torch.no_grad():
        model.instrument_projector[-1].weight.zero_()
        model.instrument_projector[-1].bias.zero_()
    z2 = model.encode_instrument(x, m)
    assert not torch.allclose(z1, z2), "projector not part of the representation"
    assert torch.allclose(z2, torch.zeros_like(z2)), "projector bypassed"


def test_random_and_pretrained_finetune_architectures_identical():
    from src.instrument_v2.train_instance_subspace_finetune import InstrumentClassifier
    from src.instrument_v2.train_sector14_matched_finetune import make_head
    shapes = {}
    for name in ("a", "b"):
        torch.manual_seed(0 if name == "a" else 99)
        backbone = build_instance_subspace("instance_mean")
        model = InstrumentClassifier(backbone, make_head(16, 0, "camccd"), True)
        shapes[name] = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert shapes["a"] == shapes["b"], "fine-tune architectures differ"


def test_training_and_selection_never_load_test_tics():
    from src.instrument_v2.group_level_dataset import Sector14ChipGroupDataset
    df = synthetic_frame(n_per_chip=8)
    all_tics = sorted(set(df["TIC"].astype(str)))
    train, test = set(all_tics[:96]), set(all_tics[96:])
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    ds = Sector14ChipGroupDataset(df, train, "shared", (t0, t1),
                                  group_size=2, return_chip=True)
    assert not (set(ds.tics) & test), "test TIC present in training dataset"
    # the supervised frame builder (used by LP-FT) hard-fails on test leakage
    from src.instrument_v2.train_sector14_matched_finetune import build_supervised_frames
    try:
        build_supervised_frames(df, train | set(list(test)[:1]), set(), test,
                                (t0, t1), "camccd")
        assert False, "test TIC accepted"
    except RuntimeError:
        pass


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
