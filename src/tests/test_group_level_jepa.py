"""Synthetic contracts for the isolated group-level instrument JEPA change."""

import numpy as np
import torch

from src.instrument_v2.group_level_dataset import Sector14ChipGroupDataset
from src.instrument_v2.group_level_jepa import GroupMeanInstrumentJEPA
from src.tests.test_sector14_jepa import synthetic_frame


def make_dataset(group_size=4):
    frame = synthetic_frame(n_per_chip=12, n_cad=64)
    t0 = min(float(np.min(time)) for time in frame["time"])
    t1 = max(float(np.max(time)) for time in frame["time"])
    return Sector14ChipGroupDataset(
        frame,
        set(frame["TIC"].astype(str)),
        "shared",
        (t0, t1),
        grid_length=64,
        group_size=group_size,
        return_chip=True,
    )


def test_sets_are_disjoint_and_same_chip():
    dataset = make_dataset()
    np.random.seed(7)
    for _ in range(300):
        context, target, chip = dataset._sample_groups()
        assert not set(context) & set(target)
        assert len(set(context)) == len(context) == dataset.group_size
        assert len(set(target)) == len(target) == dataset.group_size
        assert np.all(dataset.chips[context] == chip)
        assert np.all(dataset.chips[target] == chip)


def test_item_shapes():
    context_flux, context_mask, target_flux, target_mask, chip = make_dataset()[0]
    assert context_flux.shape == context_mask.shape == (4, 64)
    assert target_flux.shape == target_mask.shape == (4, 64)
    assert chip.ndim == 0


def test_group_mean_is_permutation_invariant_and_forward_shapes_match():
    torch.manual_seed(0)
    model = GroupMeanInstrumentJEPA(
        n_tokens=4, token_dim=4, d_model=8, n_layers=1, dropout=0.0
    ).eval()
    flux = torch.randn(2, 4, 64)
    mask = torch.ones_like(flux)
    with torch.no_grad():
        group_a = model.encode_group(flux, mask)
        permutation = torch.tensor([2, 0, 3, 1])
        group_b = model.encode_group(flux[:, permutation], mask[:, permutation])
        prediction, target, context_group, per_star = model(
            flux, mask, flux.roll(1, dims=1), mask
        )
    torch.testing.assert_close(group_a, group_b)
    assert prediction.shape == target.shape == context_group.shape == (2, 4, 4)
    assert per_star.shape == (2, 4, 4, 4)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
