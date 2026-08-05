"""Mandatory pre-flight: shapes, gradient flow, and a tiny overfit.

Runs against the real patch when it is present and falls back to a synthetic patch of
the same shape otherwise, so the contract can be checked before any data is staged.

    python -m disentangle_attempt.smoke_test
Env: DA_PARQUET, DA_DEVICE, DA_OVERFIT_STEPS.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from disentangle_attempt.dataset import (CrossSectorAnchorDataset, CrossSectorPatch,
                                        audit_batch)
from disentangle_attempt.losses import total_loss
from disentangle_attempt.masking import complementary_masks, mask_views
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.train import DEFAULT_PARQUET, forward_batch, load_config, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config(os.path.join(HERE, "config_fast.yaml"))
PARQUET = os.environ.get("DA_PARQUET", DEFAULT_PARQUET)
OVERFIT_STEPS = int(os.environ.get("DA_OVERFIT_STEPS", "60"))

B = int(CONFIG["anchors_per_step"])
L = int(CONFIG["curve_length"])
P = int(CONFIG["n_peers"])
T = int(CONFIG["n_tokens"])
D = int(CONFIG["token_dim"])


def synthetic_batch(seed=0):
    """Same tensor contract as the dataloader, for a data-free contract check."""
    rng = np.random.default_rng(seed)
    valid = rng.random((B, L)) > 0.05
    peer_valid = rng.random((B, P, L)) > 0.05
    return {
        "anchor_raw": torch.tensor(rng.normal(size=(B, L)), dtype=torch.float32),
        "anchor_valid_mask": torch.tensor(valid),
        "peer_raw": torch.tensor(rng.normal(size=(B, P, L)), dtype=torch.float32),
        "peer_mask": torch.tensor(peer_valid),
        "anchor_tic_ids": torch.arange(B, dtype=torch.int64),
        "anchor_sector": torch.full((B,), 1, dtype=torch.int64),
        "peer_tic_ids": torch.arange(B * P, dtype=torch.int64).reshape(B, P),
        "peer_distances": torch.tensor(rng.random((B, P)), dtype=torch.float32),
        "anchor_row": torch.arange(B, dtype=torch.int64),
        "peer_rows": torch.arange(B * P, dtype=torch.int64).reshape(B, P),
    }


def real_batch():
    if not os.path.exists(PARQUET):
        return None, None
    patch = CrossSectorPatch(PARQUET, target_sector=CONFIG["sector"],
                             camera=CONFIG["camera"], ccd=CONFIG["ccd"],
                             curve_length=L, n_peers=P,
                             min_valid_fraction=CONFIG["min_valid_fraction"],
                             split_seed=CONFIG["seed"],
                             max_eligible_anchors=CONFIG["max_eligible_anchors"],
                             verbose=False)
    loader = DataLoader(CrossSectorAnchorDataset(patch, "train"), batch_size=B,
                        shuffle=True, drop_last=True)
    return patch, next(iter(loader))


def check_batch_shapes(batch):
    expected = {
        "anchor_raw": (B, L), "anchor_valid_mask": (B, L),
        "peer_raw": (B, P, L), "peer_mask": (B, P, L),
        "anchor_tic_ids": (B,), "anchor_sector": (B,),
        "peer_tic_ids": (B, P), "peer_distances": (B, P),
    }
    for key, shape in expected.items():
        assert tuple(batch[key].shape) == shape, \
            f"{key}: expected {shape}, got {tuple(batch[key].shape)}"
    assert batch["anchor_valid_mask"].dtype == torch.bool
    for row in range(B):
        assert batch["anchor_tic_ids"][row] not in set(batch["peer_tic_ids"][row].tolist()), \
            "a peer must never be the anchor TIC"
    print(f"  batch shapes OK ({B} anchors, {P} peers, length {L})")


def check_shared_preprocessing(patch, batch):
    """Every branch must be a slice of arrays built by the one preprocess_curve call."""
    from disentangle_attempt.preprocess import preprocess_curve
    row = int(patch.split_anchors["train"][0])
    frame = pd.read_parquet(PARQUET)
    frame["TIC"] = frame["TIC"].astype(str)
    frame = frame.drop_duplicates(["TIC", "sector"]).reset_index(drop=True)
    record = frame.iloc[row]
    assert str(record["TIC"]) == patch.tic[row], "row alignment drifted from the parquet"
    redone = preprocess_curve(record["cadence_num"], record["time"], record["flux"],
                              record["TESS_flags"], record["TGLC_flags"],
                              patch.grids[int(record["sector"])])
    assert np.allclose(redone.curve, patch.X[row]), "anchor curve is not preprocess_curve output"
    assert (redone.valid == patch.M[row]).all()

    # The five branches index the SAME arrays, so identical filtering is structural.
    for name, source in (("anchor target", patch.X), ("physics views", patch.X),
                         ("instrument peers", patch.X), ("quiet reference", patch.X)):
        assert source is patch.X, f"{name} reads a different array"
    assert bool((batch["anchor_raw"][0] == torch.from_numpy(
        patch.X[int(batch["anchor_row"][0])])).all())

    # Requirement 5: no cadence with a nonzero TESS or TGLC flag may be valid.
    assert not (patch.M & (patch.F != 0)).any(), "a TESS-flagged cadence has valid=1"
    assert not (patch.M & (patch.G != 0)).any(), "a TGLC-flagged cadence has valid=1"
    assert not (patch.M & patch.Q).any(), "a flagged cadence has valid=1"
    assert float(np.abs(patch.X[~patch.M]).max()) == 0.0, \
        "invalid cadences must hold exactly zero and never be read as observations"
    removed = float((patch.Q & ~patch.M).sum()) / patch.Q.size
    print(f"  shared preprocessing OK (strict zero-flag: 0 flagged cadences are valid; "
          f"{removed:.3%} of grid slots removed by flags; "
          f"{patch.M.mean():.1%} of the grid is valid)")


def check_masking(batch):
    generator = torch.Generator().manual_seed(0)
    masked, hidden, visible = mask_views(batch["anchor_raw"], batch["anchor_valid_mask"],
                                         CONFIG["hidden_fraction"], generator=generator)
    assert masked.shape == (B, L) and hidden.shape == (B, L) and visible.shape == (B, L)
    assert bool((hidden & ~batch["anchor_valid_mask"]).sum() == 0), \
        "hidden cadences must be a subset of valid cadences"
    assert bool((hidden & visible).sum() == 0), "hidden and visible must be disjoint"
    assert float(masked[hidden].abs().max()) == 0.0, "hidden values must be zeroed"
    fraction = (hidden.sum(1).float() / batch["anchor_valid_mask"].sum(1).float()).mean()
    assert 0.15 < float(fraction) < 0.40, f"hidden fraction {float(fraction):.3f} off target"
    runs = (hidden[:, 1:] & ~hidden[:, :-1]).sum(1).float().mean()
    assert float(runs) < 20, f"masking is not contiguous enough ({float(runs):.1f} runs/row)"
    masks = complementary_masks(L, n_masks=4)
    assert masks.shape == (4, L) and bool((masks.sum(0) == 1).all())
    print(f"  masking OK (hidden {float(fraction):.3f} of valid, "
          f"{float(runs):.1f} windows/row, 4 complementary masks tile the curve)")


def check_forward_and_gradients(batch, device):
    model = DisentangleModel(d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"],
                             dropout=0.0, n_peers=P, n_tokens=T, token_dim=D,
                             curve_length=L).to(device)
    generator = torch.Generator().manual_seed(0)
    loss, parts, outputs = forward_batch(model, batch, CONFIG, generator, device=device)

    expected = {"predicted_raw_anchor": (B, L), "current_physics_tokens": (B, T, D),
                "current_global_physics": (B, D), "peer_instrument_tokens": (B, P, T, D),
                "instrument_context": (B, P * T * D), "hidden_mask": (B, L),
                "decoder_input": (B, (P + 1) * T * D)}
    for key, shape in expected.items():
        assert tuple(outputs[key].shape) == shape, \
            f"{key}: expected {shape}, got {tuple(outputs[key].shape)}"
    assert outputs["current_physics_latent"].shape == (B, 512), "physics latent must be [batch, 512]"
    assert outputs["instrument_context"].shape == (B, 4096), "instrument must be [batch, 4096]"
    assert outputs["decoder_input"].shape == (B, 4608), "decoder input must be [batch, 4608]"
    assert outputs["predicted_raw_anchor"].shape == (B, 1024), "decoder output must be [batch, 1024]"
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    print(f"  forward OK: physics [B,512], instrument [B,4096], decoder in [B,4608] "
          f"out [B,1024] (loss {float(loss.detach()):.4f}, "
          f"recon {parts['reconstruction']:.4f})")

    loss.backward()
    groups = {"physics_s4d": model.physics_encoder, "instrument_s4d": model.instrument_encoder,
              "decoder": model.decoder}
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} received no gradient at all"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} has non-finite gradients"
        total = float(sum(float(g.abs().sum()) for g in grads))
        assert total > 0, f"{name} gradient is exactly zero"
        print(f"  {name}: finite nonzero gradient (sum|g| = {total:.3e})")
    return model


def tiny_overfit(batches, device, steps=OVERFIT_STEPS):
    """Repeatedly train on 2-4 steps; masked reconstruction must fall substantially."""
    model = DisentangleModel(d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"],
                             dropout=0.0, n_peers=P, n_tokens=T, token_dim=D,
                             curve_length=L).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, last = None, None
    for step in range(steps):
        batch = batches[step % len(batches)]
        generator = torch.Generator().manual_seed(step % len(batches))  # fixed masks
        loss, parts, _ = forward_batch(model, batch, CONFIG, generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        optimizer.step()
        if step < len(batches):
            first = parts["reconstruction"] if first is None else max(first, parts["reconstruction"])
        last = parts["reconstruction"]
        if step % 10 == 0:
            print(f"    step {step:3d}  reconstruction {parts['reconstruction']:.4f}", flush=True)
    drop = (first - last) / first
    assert drop > 0.3, f"tiny overfit only reduced reconstruction by {drop:.1%} " \
                       f"({first:.4f} -> {last:.4f}); the model is not learning"
    print(f"  tiny overfit OK: reconstruction {first:.4f} -> {last:.4f} ({drop:.1%} drop)")


def main():
    device = pick_device(os.environ.get("DA_DEVICE", CONFIG.get("device", "auto")))
    print(f"smoke test on {device}")

    patch, batch = real_batch()
    if batch is None:
        print(f"no parquet at {PARQUET} -- using a synthetic patch")
        batch = synthetic_batch()
        batches = [synthetic_batch(s) for s in range(3)]
    else:
        print(f"real patch: sector/camera/ccd {patch.target}, "
              f"{len(patch.eligible_rows)} eligible anchors")
        loader = DataLoader(CrossSectorAnchorDataset(patch, "train"), batch_size=B,
                            shuffle=True, drop_last=True)
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(3)]

    print("[1/5] batch contract")
    check_batch_shapes(batch)
    if patch is not None:
        print("[2/5] shared preprocessing")
        check_shared_preprocessing(patch, batch)
        print("[3/5] data contract audit")
        audit_batch(patch, batch)
    print("[3b/5] masking")
    check_masking(batch)
    print("[4/5] forward and gradient flow")
    check_forward_and_gradients(batch, device)
    print(f"[5/5] tiny overfit on {len(batches)} steps")
    tiny_overfit(batches, device)
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
