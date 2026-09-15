"""Training entry points for both parts, on real or synthetic curves.

Part 1  leave-one-out peer reconstruction -> frozen systematics encoder.
Part 2  VICReg-JEPA -> physics encoder, regularised on the ONLINE latent.

Every run writes a directory holding config.json (with the git SHA and seed),
metrics.jsonl (one record per optimiser step), the best-validation checkpoint
and a final report.
"""
import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Part1Config, Part2Config
from .data import (SyntheticCurveSource, RegionGroupDataset, SingleCurveDataset,
                   random_time_mask)
from .evaluate import collapse_report, is_collapsed, mean_abs_corr, encode_source
from .losses import common_mode_loss, variance_loss, total_loss
from .models import S4Encoder, CommonModeDecoder, VICRegJEPA
from .real_data import RealCurveSource, assert_tic_disjoint, git_sha

# MPS has no complex FFT, which the S4D kernel needs -- CPU is the fallback.
DEVICE = os.environ.get(
    "TESS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ utils
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_fingerprint(module):
    """Order-stable hash of every parameter, for bit-identity assertions."""
    h = []
    for name, p in sorted(module.state_dict().items()):
        h.append(f"{name}:{hash(p.detach().cpu().numpy().tobytes())}")
    return hash("|".join(h))


class RunDir:
    def __init__(self, root, cfg, seed, extra=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.metrics_path = os.path.join(root, "metrics.jsonl")
        open(self.metrics_path, "w").close()
        payload = {"config": cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg),
                   "seed": seed, "git_sha": git_sha(), "device": DEVICE,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if extra:
            payload.update(extra)
        with open(os.path.join(root, "config.json"), "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def log(self, record):
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def path(self, name):
        return os.path.join(self.root, name)


def load_sources(args):
    """Returns {"train","val","test"} of real or synthetic sources."""
    if args.source == "real":
        srcs = {s: RealCurveSource(args.npz, s) for s in ("train", "val", "test")}
        assert_tic_disjoint(*srcs.values())
        return srcs
    syn = SyntheticCurveSource(n_regions=args.syn_regions, per_region=args.syn_per_region)
    # split synthetic by index so the same code path runs end to end
    n = len(syn.flux)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    cut_a, cut_b = int(0.65 * n), int(0.85 * n)
    parts = {"train": order[:cut_a], "val": order[cut_a:cut_b], "test": order[cut_b:]}
    out = {}
    for name, idx in parts.items():
        s = copy.copy(syn)
        s.flux, s.observed, s.region = syn.flux[idx], syn.observed[idx], syn.region[idx]
        s.split_name = name
        s.tic = np.array([f"SYN{i}" for i in idx])
        s.area = s.region
        s.camera = np.zeros(len(idx), dtype=np.int16)
        s.ccd = np.zeros(len(idx), dtype=np.int16)
        s.sector = np.zeros(len(idx), dtype=np.int16)
        s.label = np.full(len(idx), -1, dtype=np.int16)
        out[name] = s
    return out


# ------------------------------------------------------------------ part 1
def _part1_epoch_loss(enc, dec, loader, cfg, device, max_batches=None):
    enc.eval()
    dec.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b, (flux, observed) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            flux, observed = flux.to(device), observed.to(device)
            B, G, L = flux.shape
            z = enc(flux.reshape(B * G, L), observed.reshape(B * G, L)).reshape(B, G, -1)
            tot += float(common_mode_loss(dec, z, flux, observed))
            n += 1
    enc.train()
    dec.train()
    return tot / max(n, 1)


def region_median_baseline(loader, max_batches=20):
    """Reconstruct each curve with the median of its 31 peers. Part 1 must beat this."""
    tot, n = 0.0, 0
    for b, (flux, observed) in enumerate(loader):
        if b >= max_batches:
            break
        f, o = flux.numpy(), observed.numpy()
        B, G, L = f.shape
        for i in range(B):
            for j in range(G):
                peers = np.delete(np.arange(G), j)
                w = o[i, peers]
                stack = np.where(w > 0, f[i, peers], np.nan)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pred = np.nanmedian(stack, axis=0)
                pred = np.nan_to_num(pred)
                m = o[i, j]
                tot += float((((pred - f[i, j]) ** 2) * m).sum() / max(m.sum(), 1))
                n += 1
    return tot / max(n, 1)


def train_part1(sources, cfg=Part1Config(), seed=0, run_root=None, device=DEVICE):
    set_seed(seed)
    run = RunDir(run_root or "artifacts/vicreg_jepa/part1", cfg, seed,
                 {"part": 1, "splits": {k: getattr(v, "describe", lambda: len(v.flux))()
                                        for k, v in sources.items()}})

    tr = DataLoader(RegionGroupDataset(sources["train"], cfg.group_size, seed=seed),
                    batch_size=cfg.batch_groups, num_workers=0, drop_last=True)
    va = DataLoader(RegionGroupDataset(sources["val"], cfg.group_size,
                                       groups_per_epoch=cfg.val_groups, seed=seed + 1),
                    batch_size=cfg.batch_groups, num_workers=0, drop_last=True)

    enc = S4Encoder(cfg.latent_dim, cfg.n_tokens, cfg.d_model,
                    cfg.d_state, cfg.n_layers, cfg.dropout).to(device)
    dec = CommonModeDecoder(cfg.latent_dim, cfg.seq_len).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    step = 0
    while step < cfg.steps:
        for flux, observed in tr:
            flux, observed = flux.to(device), observed.to(device)
            B, G, L = flux.shape
            z = enc(flux.reshape(B * G, L), observed.reshape(B * G, L)).reshape(B, G, -1)
            l_common = common_mode_loss(dec, z, flux, observed)
            l_var = variance_loss(z.reshape(B * G, -1), cfg.var_gamma)
            loss = l_common + cfg.var_weight * l_var

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), cfg.grad_clip)
            opt.step()

            rec = {"step": step, "train_common": l_common.item(),
                   "train_var": l_var.item(), "grad_norm": float(gnorm)}
            if step % cfg.eval_every == 0 or step == cfg.steps - 1:
                val = _part1_epoch_loss(enc, dec, va, cfg, device, cfg.val_batches)
                flat = z.reshape(B * G, -1).detach().cpu().numpy()
                rec.update({"val_common": val, **{f"latent_{k}": v
                                                  for k, v in collapse_report(flat).items()}})
                if val < best:
                    best = val
                    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                                "cfg": cfg.__dict__, "seed": seed, "step": step,
                                "val_common": val, "git_sha": git_sha()},
                               run.path("part1_best.pt"))
                    rec["saved_best"] = True
                print(f"[part1] {step:6d} train={l_common.item():.4f} val={val:.4f} "
                      f"var={float(l_var):.4f} best={best:.4f}", flush=True)
            run.log(rec)
            step += 1
            if step >= cfg.steps:
                break

    baseline = region_median_baseline(va, max_batches=cfg.val_batches)
    summary = {"best_val_common": best, "region_median_baseline": baseline,
               "beats_baseline": bool(best < baseline)}
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[part1] best val {best:.4f} vs region-median baseline {baseline:.4f} "
          f"-> {'BEATS' if best < baseline else 'DOES NOT BEAT'}", flush=True)
    return run.path("part1_best.pt"), summary


def load_frozen_part1(ckpt_path, device=DEVICE):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    enc = S4Encoder(c["latent_dim"], c["n_tokens"], c["d_model"],
                    c["d_state"], c["n_layers"], 0.0)
    enc.load_state_dict(blob["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device)


# ------------------------------------------------------------------ part 2
def verify_checklist(model, batch, cfg):
    """Pre-training checklist, run against one real batch."""
    flux, observed, visible = batch
    assert not any(p.requires_grad for p in model.part1.parameters()), "Part 1 not frozen"
    assert not any(p.requires_grad for p in model.ema_encoder.parameters()), "EMA not frozen"
    assert not model.part1.training, "Part 1 not in eval mode"

    out = model(flux, observed, visible)
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

    # EMA starts as an exact copy, so tau*t + (1-tau)*t = t and it cannot move
    # until the online weights do. Stand in for a step, check the rule, roll back.
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


@torch.no_grad()
def _part2_val(model, sources, cfg, device, n=1024):
    """Validation L_inv plus every collapse / disentanglement diagnostic."""
    model.eval()
    src = sources["val"]
    flux = torch.from_numpy(np.asarray(src.flux[:n], dtype=np.float32)).to(device)
    obs = torch.from_numpy(np.asarray(src.observed[:n], dtype=np.float32)).to(device)
    inv_tot, zs_all, zm_all, batches = 0.0, [], [], 0
    for i in range(0, len(flux), cfg.batch_size):
        f, o = flux[i:i + cfg.batch_size], obs[i:i + cfg.batch_size]
        if len(f) < 8:
            break
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        zm, zu, zsy, zp = model(f, o, v)
        inv_tot += float(torch.nn.functional.mse_loss(zp, zu))
        zm_all.append(zm.cpu().numpy())
        zs_all.append(zsy.cpu().numpy())
        batches += 1
    model.train()
    zm_all = np.concatenate(zm_all)
    zs_all = np.concatenate(zs_all)
    rep = collapse_report(zm_all)
    rep["val_inv"] = inv_tot / max(batches, 1)
    rep["val_mean_abs_corr"] = mean_abs_corr(zm_all, zs_all)
    rep["collapsed"] = is_collapsed(rep)
    return rep


def train_part2(sources, part1_ckpt, cfg=Part2Config(), seed=0, run_root=None,
                device=DEVICE, random_init_part1=False):
    set_seed(seed)
    run = RunDir(run_root or "artifacts/vicreg_jepa/part2", cfg, seed,
                 {"part": 2, "part1_ckpt": part1_ckpt,
                  "random_init_part1": random_init_part1})

    if random_init_part1:
        p1 = S4Encoder(cfg.sys_dim, cfg.n_tokens, cfg.d_model,
                       cfg.d_state, cfg.n_layers, 0.0).to(device)
        p1.eval()
        for p in p1.parameters():
            p.requires_grad = False
    else:
        p1 = load_frozen_part1(part1_ckpt, device)

    p1_fingerprint = param_fingerprint(p1)
    model = VICRegJEPA(p1, cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    dl = DataLoader(SingleCurveDataset(sources["train"]), batch_size=cfg.batch_size,
                    shuffle=True, drop_last=True)

    best, best_step, checked, step = float("inf"), -1, False, 0
    while step < cfg.steps:
        for flux, observed in dl:
            flux, observed = flux.to(device), observed.to(device)
            visible = random_time_mask(flux, observed,
                                       cfg.mask_ratio_min, cfg.mask_ratio_max)
            if not checked:
                verify_checklist(model, (flux, observed, visible), cfg)
                checked = True

            zm, zu, zs, zp = model(flux, observed, visible)
            loss, parts = total_loss(zm, zu, zs, zp, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
            model.update_ema(cfg.tau)

            std = zm.detach().std(dim=0)
            rec = {"step": step, **parts,
                   "std_min": float(std.min()), "std_median": float(std.median()),
                   "frac_std_below_0.5": float((std < 0.5).float().mean()),
                   "grad_norm": float(gnorm)}

            if step % cfg.eval_every == 0 or step == cfg.steps - 1:
                v = _part2_val(model, sources, cfg, device)
                rec.update({f"val_{k}": val for k, val in v.items()})
                if v["val_inv"] < best and not v["collapsed"]:
                    best, best_step = v["val_inv"], step
                    torch.save({"encoder": model.encoder.state_dict(),
                                "ema_encoder": model.ema_encoder.state_dict(),
                                "predictor": model.predictor.state_dict(),
                                "cfg": cfg.__dict__, "seed": seed, "step": step,
                                "val": v, "git_sha": git_sha()}, run.path("part2_best.pt"))
                    rec["saved_best"] = True
                print(f"[part2] {step:6d} inv={parts['inv']:.4f} cor={parts['cor_sys']:.4f} "
                      f"var={parts['var']:.4f} cov={parts['cov']:.4f} | "
                      f"val_inv={v['val_inv']:.4f} std_med={v['std_median']:.3f} "
                      f"erank={v['effective_rank']:.1f} "
                      f"{'COLLAPSED' if v['collapsed'] else ''}", flush=True)
            run.log(rec)
            step += 1
            if step >= cfg.steps:
                break

    assert param_fingerprint(model.part1) == p1_fingerprint, \
        "frozen Part 1 parameters changed during Part 2"
    final = _part2_val(model, sources, cfg, device)
    summary = {"best_val_inv": best, "best_step": best_step,
               "final": final, "part1_unchanged": True}
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[part2] best val_inv {best:.4f} @ step {best_step} | "
          f"final collapsed={final['collapsed']}", flush=True)
    return run.path("part2_best.pt"), summary, model


# ------------------------------------------------------------------ cli
def add_common_args(ap):
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--npz", default="artifacts/vicreg_jepa/curves.npz")
    ap.add_argument("--syn-regions", type=int, default=16)
    ap.add_argument("--syn-per-region", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/vicreg_jepa/run")
    ap.add_argument("--steps1", type=int, default=None)
    ap.add_argument("--steps2", type=int, default=None)
    ap.add_argument("--small", action="store_true", help="shrink the net for CPU")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    return ap


def configs_from_args(args):
    c1, c2 = Part1Config(), Part2Config()
    if args.small:
        for c in (c1, c2):
            c.d_model, c.d_state, c.n_layers = 64, 16, 2
        c1.batch_groups, c2.batch_size = 2, 64
    if args.steps1:
        c1.steps = args.steps1
    if args.steps2:
        c2.steps = args.steps2
    if getattr(args, "eval_every", None):
        c1.eval_every = c2.eval_every = args.eval_every
    if getattr(args, "val_batches", None):
        c1.val_batches = args.val_batches
    if getattr(args, "batch_size", None):
        c2.batch_size = args.batch_size
    return c1, c2


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--part", choices=["1", "2", "both"], default="both")
    ap.add_argument("--part1-ckpt", default=None)
    args = ap.parse_args()

    sources = load_sources(args)
    c1, c2 = configs_from_args(args)
    print(f"device={DEVICE} seed={args.seed} source={args.source} "
          f"sizes={ {k: len(v.flux) for k, v in sources.items()} }", flush=True)

    ckpt = args.part1_ckpt
    if args.part in ("1", "both"):
        ckpt, _ = train_part1(sources, c1, args.seed, os.path.join(args.out, "part1"))
    if args.part in ("2", "both"):
        train_part2(sources, ckpt, c2, args.seed, os.path.join(args.out, "part2"))
