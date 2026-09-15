import argparse
import copy
import os
import torch
from torch.utils.data import DataLoader

from .config import Part1Config, Part2Config
from .data import (SyntheticCurveSource, NpzCurveSource, RegionGroupDataset,
                   SingleCurveDataset, random_time_mask)
from .models import S4Encoder, CommonModeDecoder, VICRegJEPA
from .losses import common_mode_loss, variance_loss, total_loss

# MPS has no complex FFT, which the S4D kernel needs -- CPU is the fallback.
DEVICE = os.environ.get(
    "TESS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def train_part1(source, cfg=Part1Config()):
    ds = RegionGroupDataset(source, cfg.group_size)
    dl = DataLoader(ds, batch_size=cfg.batch_groups, num_workers=0, drop_last=True)

    enc = S4Encoder(cfg.latent_dim, cfg.n_tokens, cfg.d_model,
                    cfg.d_state, cfg.n_layers, cfg.dropout).to(DEVICE)
    dec = CommonModeDecoder(cfg.latent_dim, cfg.seq_len).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    while step < cfg.steps:
        for flux, observed in dl:
            flux, observed = flux.to(DEVICE), observed.to(DEVICE)
            B, G, L = flux.shape
            z = enc(flux.reshape(B * G, L), observed.reshape(B * G, L)).reshape(B, G, -1)

            l_common = common_mode_loss(dec, z, flux, observed)
            l_var = variance_loss(z.reshape(B * G, -1), cfg.var_gamma)
            loss = l_common + cfg.var_weight * l_var

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                print(f"[part1] {step:6d}  common={l_common.item():.4f}  "
                      f"var={l_var.item():.4f}", flush=True)
            step += 1
            if step >= cfg.steps:
                break

    os.makedirs(os.path.dirname(cfg.ckpt) or ".", exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.__dict__}, cfg.ckpt)
    print(f"[part1] saved -> {cfg.ckpt}")
    return enc


def load_frozen_part1(cfg1=Part1Config()):
    enc = S4Encoder(cfg1.latent_dim, cfg1.n_tokens, cfg1.d_model,
                    cfg1.d_state, cfg1.n_layers, 0.0)
    enc.load_state_dict(torch.load(cfg1.ckpt, map_location="cpu")["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def verify_checklist(model, batch, cfg):
    """The pre-training checklist, run against one real batch."""
    flux, observed, visible = batch
    assert not any(p.requires_grad for p in model.part1.parameters()), "Part 1 not frozen"
    assert not any(p.requires_grad for p in model.ema_encoder.parameters()), "EMA not frozen"
    assert not model.part1.training, "Part 1 not in eval mode"

    out = model(flux, observed, visible)                     # shape asserts inside
    assert all(torch.isfinite(t).all() for t in out), "NaN/inf in forward"
    loss, parts = total_loss(*out, cfg)
    assert torch.isfinite(loss), "NaN in total loss"

    loss.backward()
    for name, mod in (("encoder", model.encoder), ("predictor", model.predictor)):
        g = [p.grad for p in mod.parameters() if p.grad is not None]
        assert g and max(x.abs().max().item() for x in g) > 0, f"no gradient into {name}"
    assert all(p.grad is None for p in model.ema_encoder.parameters()), "grad leaked into EMA"
    assert all(p.grad is None for p in model.part1.parameters()), "grad leaked into Part 1"
    model.zero_grad(set_to_none=True)

    # The EMA encoder starts as an exact copy, so tau*theta + (1-tau)*theta = theta
    # and it cannot move until the online weights do. Stand in for an optimiser
    # step, check the update rule lands where Instruction 2.9 says, then roll back.
    online_state = copy.deepcopy(model.encoder.state_dict())
    ema_state = copy.deepcopy(model.ema_encoder.state_dict())
    delta, ref = 1e-2, next(model.ema_encoder.parameters()).clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.full_like(p, delta))
    model.update_ema(cfg.tau)
    moved = (next(model.ema_encoder.parameters()) - ref).abs().max().item()
    expected = (1 - cfg.tau) * delta
    assert abs(moved - expected) <= 0.05 * expected, \
        f"EMA rule wrong: moved {moved:.3e}, expected {expected:.3e}"
    model.encoder.load_state_dict(online_state)
    model.ema_encoder.load_state_dict(ema_state)

    print(f"[check] all passed | {parts} | ema step {moved:.2e} "
          f"(expected {expected:.2e})", flush=True)


def train_part2(source, part1_encoder, cfg=Part2Config()):
    ds = SingleCurveDataset(source)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = VICRegJEPA(part1_encoder, cfg).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    checked, step = False, 0
    while step < cfg.steps:
        for flux, observed in dl:
            flux, observed = flux.to(DEVICE), observed.to(DEVICE)
            # Instruction 2.2 -- fresh mask every iteration
            visible = random_time_mask(flux, observed,
                                       cfg.mask_ratio_min, cfg.mask_ratio_max)
            if not checked:
                verify_checklist(model, (flux, observed, visible), cfg)
                checked = True

            z_masked, z_unmasked, z_sys, z_pred = model(flux, observed, visible)
            loss, parts = total_loss(z_masked, z_unmasked, z_sys, z_pred, cfg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_ema(cfg.tau)                        # Instruction 2.9

            if step % cfg.log_every == 0:
                std = z_masked.std(dim=0)
                print(f"[part2] {step:6d}  inv={parts['inv']:.4f}  "
                      f"cor_sys={parts['cor_sys']:.4f}  var={parts['var']:.4f}  "
                      f"cov={parts['cov']:.4f}  total={parts['total']:.4f}  "
                      f"| std min/med {std.min():.3f}/{std.median():.3f}", flush=True)
            step += 1
            if step >= cfg.steps:
                break

    os.makedirs(os.path.dirname(cfg.ckpt) or ".", exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(), "cfg": cfg.__dict__}, cfg.ckpt)
    print(f"[part2] saved -> {cfg.ckpt}")
    return model


def smoke_configs():
    """Small enough to run both parts on CPU in well under a minute."""
    c1, c2 = Part1Config(), Part2Config()
    for c in (c1, c2):
        c.d_model, c.d_state, c.n_layers = 64, 16, 2
        c.steps, c.log_every = 20, 5
    c1.batch_groups = 2
    c2.batch_size = 64
    return c1, c2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="flux/observed/region arrays")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        source = SyntheticCurveSource(n_regions=8, per_region=40)
        c1, c2 = smoke_configs()
    else:
        source = NpzCurveSource(args.npz) if args.npz else SyntheticCurveSource()
        c1, c2 = Part1Config(), Part2Config()

    print(f"device={DEVICE}  curves={len(source.flux)}", flush=True)
    train_part1(source, c1)
    train_part2(source, load_frozen_part1(c1), c2)
