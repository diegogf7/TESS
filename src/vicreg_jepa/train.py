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
from .data import (SyntheticCurveSource, RegionGroupDataset, LocalGroupDataset,
                   SingleCurveDataset, random_time_mask)
from .collapse import CollapseThresholds, assess
from .evaluate import (collapse_report, collapse_reasons, is_collapsed,
                       mean_abs_corr, encode_source)
from .losses import (common_mode_loss, variance_loss, total_loss,
                     zero_baseline_loss)
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


class ValidationProbe:
    """Fixed validation examples and fixed masking seeds, shared by every trial.

    Requirement 4 of the search specification: two trials must be judged on the
    same curves under the same mask draws, so that a difference in their
    diagnostics is a difference in the model and not in the sample.
    """

    def __init__(self, source, n=1024, batch_size=256, seed=1234, device="cpu"):
        idx = np.random.default_rng(seed).permutation(len(source.flux))[:n]
        idx.sort()
        self.index = idx
        self.flux = torch.from_numpy(np.asarray(source.flux[idx], dtype=np.float32))
        self.observed = torch.from_numpy(np.asarray(source.observed[idx], dtype=np.float32))
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def batches(self, cfg):
        """Deterministic (flux, observed, visible) batches for one validation pass."""
        gen = torch.Generator().manual_seed(self.seed)
        for i in range(0, len(self.flux), self.batch_size):
            f = self.flux[i:i + self.batch_size].to(self.device)
            o = self.observed[i:i + self.batch_size].to(self.device)
            if len(f) < 8:
                break
            v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max,
                                 generator=gen)
            yield f, o, v


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

    def groups(src, per_epoch, sd):
        if getattr(cfg, "local_groups", False) and hasattr(src, "ra"):
            ds = LocalGroupDataset(src, cfg.group_size, per_epoch, sd,
                                   cfg.group_radius_deg)
            print(f"[part1] local groups ({src.split_name}): {ds.stats()}", flush=True)
            return ds
        return RegionGroupDataset(src, cfg.group_size, per_epoch, sd)

    tr = DataLoader(groups(sources["train"], 2000, seed),
                    batch_size=cfg.batch_groups, num_workers=0, drop_last=True)
    va = DataLoader(groups(sources["val"], cfg.val_groups, seed + 1),
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
    zeros, nb = 0.0, 0
    for b, (fl, ob) in enumerate(va):
        if b >= cfg.val_batches:
            break
        zeros += float(zero_baseline_loss(fl.to(device), ob.to(device)))
        nb += 1
    zeros /= max(nb, 1)
    summary = {"best_val_common": best,
               "region_median_baseline": baseline,
               "zero_prediction_baseline": zeros,
               "beats_region_median": bool(best < baseline),
               "beats_zero_prediction": bool(best < zeros),
               "gain_over_zero_pct": 100 * (1 - best / zeros) if zeros else None}
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[part1] best val {best:.4f} | zero-prediction {zeros:.4f} "
          f"({summary['gain_over_zero_pct']:+.1f}%) | region-median {baseline:.4f} "
          f"-> {'BEATS zero' if best < zeros else 'DOES NOT BEAT zero'}", flush=True)
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
def _part2_val(model, probe, cfg, device, thresholds, online_fail_streak=0):
    """Validation on the fixed probe, measuring all three representations."""
    model.eval()
    inv_tot, batches = 0.0, 0
    lanes = {"masked_online": [], "full_online": [], "full_ema": []}
    sysl = []
    for f, o, v in probe.batches(cfg):
        zm, zu, zsy, zp = model(f, o, v)
        inv_tot += float(torch.nn.functional.mse_loss(zp, zu))
        lanes["masked_online"].append(zm.cpu().numpy())
        lanes["full_ema"].append(zu.cpu().numpy())
        lanes["full_online"].append(model.encoder(f, o, pool_mask=o).cpu().numpy())
        sysl.append(zsy.cpu().numpy())
        batches += 1
    model.train()

    lanes = {k: np.concatenate(v) for k, v in lanes.items()}
    sysl = np.concatenate(sysl)
    verdict = assess(lanes, thresholds, online_fail_streak)
    ema = verdict["stats"]["full_ema"]
    return {
        "val_inv": inv_tot / max(batches, 1),
        "val_mean_abs_corr": mean_abs_corr(lanes["full_ema"], sysl),
        "rejected": verdict["rejected"],
        "why": verdict["why"],
        "online_fail_streak": verdict["online_fail_streak"],
        "meets_target_rank": verdict["meets_target_rank"],
        "effective_rank": ema["effective_rank"],
        "std_median": ema["std_median"],
        "frac_std_below_floor": ema["frac_std_below_floor"],
        "offdiag_cov_rms": ema["offdiag_cov_rms"],
        "dup_pair_frac": ema["dup_pair_frac"],
        "lanes": {k: verdict["stats"][k] for k in lanes},
        "lane_reasons": verdict["reasons"],
    }


def train_part2(sources, part1_ckpt, cfg=Part2Config(), seed=0, run_root=None,
                device=DEVICE, random_init_part1=False, thresholds=None,
                probe=None, warmup_frac=0.10, prune_after=None):
    """Train Part 2, saving ONLY checkpoints that pass every collapse check.

    Returns (ckpt_or_None, summary, model). summary["status"] is one of
    "ok" (a valid checkpoint exists), "failed_all_collapsed" (none ever passed),
    or "pruned" (collapsed for `prune_after` consecutive validations).
    """
    set_seed(seed)
    thresholds = thresholds or CollapseThresholds()
    run = RunDir(run_root or "artifacts/vicreg_jepa/part2", cfg, seed,
                 {"part": 2, "part1_ckpt": part1_ckpt,
                  "random_init_part1": random_init_part1,
                  "thresholds": thresholds.to_dict()})

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
    probe = probe or ValidationProbe(sources["val"], batch_size=cfg.batch_size,
                                     device=device)

    best = float("inf")
    best_step, best_val = -1, None
    checked, step = False, 0
    streak, reject_run = 0, 0
    warmup_steps = int(warmup_frac * cfg.steps)
    collapse_free_after_warmup = True
    status = "ok"

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
                v = _part2_val(model, probe, cfg, device, thresholds, streak)
                streak = v["online_fail_streak"]
                rec.update({f"val_{k}": val for k, val in v.items()
                            if k not in ("lanes", "lane_reasons")})
                rec["val_lanes"] = v["lanes"]

                if v["rejected"]:
                    reject_run += 1
                    if step >= warmup_steps:
                        collapse_free_after_warmup = False
                else:
                    reject_run = 0
                    # Requirement 2: save ONLY when every collapse check passes.
                    if v["val_inv"] < best:
                        best, best_step, best_val = v["val_inv"], step, v
                        torch.save({"encoder": model.encoder.state_dict(),
                                    "ema_encoder": model.ema_encoder.state_dict(),
                                    "predictor": model.predictor.state_dict(),
                                    "cfg": cfg.__dict__, "seed": seed, "step": step,
                                    "val": v, "thresholds": thresholds.to_dict(),
                                    "git_sha": git_sha()}, run.path("part2_best.pt"))
                        rec["saved_best"] = True

                print(f"[part2] {step:6d} inv={parts['inv']:.4f} cor={parts['cor_sys']:.4f} "
                      f"| val_inv={v['val_inv']:.4f} erank={v['effective_rank']:.1f} "
                      f"std_med={v['std_median']:.3f} dup={v['dup_pair_frac']:.2f}"
                      + (f"  REJECT {'; '.join(v['why'][:2])}" if v["rejected"] else "  ok"),
                      flush=True)

                if prune_after and reject_run >= prune_after:
                    status = "pruned"
                    print(f"[part2] pruned after {reject_run} consecutive rejected "
                          f"validations", flush=True)
                    run.log(rec)
                    step = cfg.steps
                    break

            run.log(rec)
            step += 1
            if step >= cfg.steps:
                break

    assert param_fingerprint(model.part1) == p1_fingerprint, \
        "frozen Part 1 parameters changed during Part 2"

    if best_step < 0 and status != "pruned":
        status = "failed_all_collapsed"

    ckpt = run.path("part2_best.pt") if best_step >= 0 else None
    summary = {
        "status": status,
        "best_val_inv": best if best_step >= 0 else None,
        "best_step": best_step,
        "best_val": best_val,
        "collapse_free_after_warmup": collapse_free_after_warmup,
        "meets_target_rank": bool(best_val and best_val["meets_target_rank"]),
        "part1_unchanged": True,
        "checkpoint": ckpt,
    }
    with open(run.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if ckpt:
        print(f"[part2] status={status} best val_inv {best:.4f} @ step {best_step} "
              f"erank={best_val['effective_rank']:.1f} "
              f"target_rank={'MET' if summary['meets_target_rank'] else 'not met'} "
              f"collapse_free={collapse_free_after_warmup}", flush=True)
    else:
        print(f"[part2] status={status} -- NO VALID CHECKPOINT, trial failed", flush=True)
    return ckpt, summary, model


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
    ap.add_argument("--group-radius", type=float, default=None,
                    help="cap on a Part 1 group's angular radius, degrees")
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
    if getattr(args, "group_radius", None):
        c1.group_radius_deg = args.group_radius
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
