# All this code is from Claude
"""Two-stage custom group-level TGLC CBV JEPA, K=8, MLP predictors (NO Transformer).

METADATA-GUIDED (area/region membership selects each star's basis). Only the
teacher-target construction changes vs the existing regional-teacher pipeline;
data, splits, cadence grid, S4D, MLP predictor, EMA, JEPA loss, and probe are
reused unchanged.

  Stage A  regional teacher: two disjoint 8-star same-area groups
             -> median+log-MAD -> K=8 CBV reconstruction -> online/EMA S4D + MLP
             -> gap-blind JEPA loss (EMA update). Select best val AREA bacc.
  Freeze   selected Stage-A EMA S4D -> Stage-B frozen teacher (hash-verified).
  Stage B  single-star student: raw context star -> online S4D + MLP -> JEPA loss
             vs frozen teacher on the 8-star CBV reconstruction (context excluded).
             Train online S4D + MLP only; evaluate the frozen online encoder.

    python -m src.instrument_v2.train_group_cbv_k8_mlp
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
    group_statistics,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA,
    fixed_teacher_loss,
    load_frozen_teacher,
)
from src.instrument_v2.regional_group_teacher import (
    AreaGroupPairDataset,
    build_regional_teacher,
    state_hash,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import effective_rank, git_commit, seed_worker
from src.loss_function.gapblind_fix import gapblind_loss
from src.instrument_v2.regional_cbv import (
    build_or_load_area_bases,
    cbv_fingerprint,
    ridge_reconstruct,
)

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
EPOCHS = min(int(os.environ.get("EPOCHS", "15")), 15)
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "8"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
BATCH = int(os.environ.get("BATCH", "64"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-2"))
GROUP_MIN_VALID = int(os.environ.get("GROUP_MIN_VALID", "4"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
N_PROBE_DRAWS = int(os.environ.get("N_PROBE_DRAWS", "6"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
CBV_ART_DIR = os.environ.get(
    "CBV_ART_DIR", os.path.join("artifacts", "instrument_v2", "cbv_refinement_screen"))
ART_DIR = os.environ.get(
    "GROUP_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group_cbv_k8_mlp_v1"))
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group_cbv_k8_mlp_v1")


def stack_stats(median, log_mad):
    return torch.stack([median, log_mad], dim=-1)


# ---------------------------------------------------------- CBV datasets
class AreaGroupCBVPairDataset(AreaGroupPairDataset):
    """Stage A: two disjoint 8-star same-area groups; each group's median is
    replaced by its K=8 CBV reconstruction before stacking with log-MAD."""

    def __init__(self, base, area_bases, ridge_lambda, items_per_epoch=None):
        super().__init__(base, items_per_epoch)
        self.area_bases = area_bases
        self.ridge_lambda = ridge_lambda
        self.eligible = [a for a in self.eligible if a in area_bases]
        if not self.eligible:
            raise RuntimeError("no area has >= 2K stars AND a CBV basis")

    def group_input(self, rows):
        area = int(self.base.group_labels[rows[0]])
        instrument, log_mad, valid = cbv_fingerprint(
            self.base.X, self.base.M, rows, self.area_bases[area],
            self.base.min_valid, self.ridge_lambda)
        stats = torch.stack([torch.tensor(instrument), torch.tensor(log_mad)], dim=-1)
        return stats, torch.tensor(valid)

    def draw_groups(self):
        area = self.eligible[np.random.randint(len(self.eligible))]
        rows = np.random.choice(self.base.group_rows[area], size=2 * self.base.k,
                                replace=False)
        return rows[:self.base.k], rows[self.base.k:], area

    def __getitem__(self, idx):
        rows_a, rows_b, area = self.draw_groups()
        stats_a, valid_a = self.group_input(rows_a)
        stats_b, valid_b = self.group_input(rows_b)
        return (stats_a, valid_a, stats_b, valid_b,
                torch.tensor(int(area), dtype=torch.int64),
                torch.tensor(int(area) // 10, dtype=torch.int64))


class Sector14GroupCBVReconDataset(Sector14GroupStatDataset):
    """Stage B: same sampling as the parent (context excluded from its 8 teacher
    stars), but the teacher median channel is the 8-star median's K=8 masked-
    ridge CBV reconstruction. Context input and .X/.chips stay RAW for probing."""

    def __init__(self, *args, area_bases=None, ridge_lambda=RIDGE_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        if area_bases is None:
            raise ValueError("need train-only per-area CBV bases")
        self.area_bases = area_bases
        self.ridge_lambda = ridge_lambda
        self.group_rows = {g: r for g, r in self.group_rows.items() if g in area_bases}
        self.group_list = sorted(self.group_rows)
        if not self.group_list:
            raise RuntimeError("no area has both >= K+1 stars and a CBV basis")

    def __getitem__(self, idx):
        context, targets, group = self._sample_item()
        median, log_mad, valid, n_observed = group_statistics(
            self.X[targets], self.M[targets], self.min_valid)
        instrument = ridge_reconstruct(median, valid, self.area_bases[int(group)],
                                       self.ridge_lambda)
        return (torch.tensor(self.X[context]), torch.tensor(self.M[context]),
                torch.tensor(instrument), torch.tensor(log_mad),
                torch.tensor(valid), torch.tensor(n_observed),
                torch.tensor(group, dtype=torch.int64))


# --------------------------------------------------------------- Stage A
def run_epoch_a(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            stats_a, valid_a, stats_b, valid_b, _, _ = [t.to(DEVICE) for t in batch]
            prediction, target, context_tokens = model(stats_a, valid_a, stats_b, valid_b)
            loss = gapblind_loss(prediction, target, context_tokens,
                                 target_mask=valid_b, var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()          # existing EMA update
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def group_features(model, pair_dataset, view="online"):
    features, areas, chips = [], [], []
    base = pair_dataset.base
    with torch.no_grad():
        for area in pair_dataset.eligible:
            for _ in range(N_PROBE_DRAWS):
                rows = np.random.choice(base.group_rows[area], size=base.k, replace=False)
                stats, valid = pair_dataset.group_input(rows)
                tokens = model.encode(stats.unsqueeze(0).to(DEVICE),
                                      valid.unsqueeze(0).to(DEVICE), view=view)
                features.append(tokens.flatten(1).cpu().numpy()[0])
                areas.append(int(area))
                chips.append(int(area) // 10)
    return np.asarray(features), np.asarray(areas), np.asarray(chips)


def train_stage_a(df, train_tics, val_tics, test_tics, t_range, bases):
    tag = f"regteacher_cbv_k{K}_s{SEED}"
    ckpt = os.path.join(CKPT_DIR, f"{tag}_best.pth")
    train_pairs = AreaGroupCBVPairDataset(
        Sector14GroupStatDataset(df, train_tics, t_range, "area", K), bases, RIDGE_LAMBDA)
    val_pairs = AreaGroupCBVPairDataset(
        Sector14GroupStatDataset(df, val_tics, t_range, "area", K), bases, RIDGE_LAMBDA)
    assert not (set(train_pairs.base.tics) | set(val_pairs.base.tics)) & test_tics
    train_loader = DataLoader(train_pairs, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED), drop_last=True)
    val_loader = DataLoader(val_pairs, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED), drop_last=True)

    model = build_regional_teacher().to(DEVICE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ema_hash0 = state_hash(model.ema_encoder)

    fields = ["epoch", "train_loss", "val_loss", "val_area_bacc",
              "val_camccd_bacc", "effective_rank", "ema_moving"]
    metrics_path = os.path.join(ART_DIR, f"metrics_stageA_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"area_bacc": -1.0, "epoch": 0, "camccd_bacc": float("nan"),
            "erank": float("nan")}
    since_best = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_a(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch_a(model, val_loader)
        train_z, train_area, train_chip = group_features(model, train_pairs)
        val_z, val_area, val_chip = group_features(model, val_pairs)
        erank = effective_rank(val_z)
        area_bacc = fast_probe(train_z, train_area, val_z, val_area)
        camccd = fast_probe(train_z, train_chip, val_z, val_chip)
        ema_moving = state_hash(model.ema_encoder) != ema_hash0
        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_area_bacc": area_bacc, "val_camccd_bacc": camccd,
                 "effective_rank": erank, "ema_moving": ema_moving})
        marker = ""
        if area_bacc > best["area_bacc"]:
            best = {"area_bacc": area_bacc, "epoch": epoch, "camccd_bacc": camccd,
                    "erank": erank}
            torch.save(model.state_dict(), ckpt)
            since_best = 0
            marker = " <- best"
        else:
            since_best += 1
        print(f"[A] epoch {epoch:02d}: train={train_loss:.5f} val={val_loss:.5f} "
              f"area={area_bacc:.4f} camccd={camccd:.4f} erank={erank:.1f} "
              f"ema_moving={ema_moving}{marker}", flush=True)
        if epoch >= MIN_EPOCHS and since_best >= PATIENCE:
            print(f"[A] early stop at epoch {epoch}", flush=True)
            break

    selection = {"tag": tag, "seed": SEED, "k": K, "ridge_lambda": RIDGE_LAMBDA,
                 "best": best, "checkpoint": ckpt, "git_commit": git_commit()}
    sel_path = os.path.join(ART_DIR, f"selection_{tag}.json")
    with open(sel_path, "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    return sel_path, best


# --------------------------------------------------------------- Stage B
def run_epoch_b(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m, median, log_mad, valid, _, _ = [t.to(DEVICE) for t in batch]
            prediction, target, tokens = model(
                ctx_f, ctx_m, stack_stats(median, log_mad), valid)
            loss = fixed_teacher_loss(prediction, target, tokens, valid, var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()               # NO EMA: teacher frozen
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def probe_b(model, train_ds, val_ds):
    tr = individual_latents(model, train_ds, "online")
    va = individual_latents(model, val_ds, "online")
    tc, vc = train_ds.chips, val_ds.chips
    return (fast_probe(tr, tc, va, vc), fast_probe(tr, tc // 4, va, vc // 4),
            fast_probe(tr, tc % 4, va, vc % 4), effective_rank(va))


def train_stage_b(df, train_tics, val_tics, test_tics, t_range, bases, teacher_sel_path):
    tag = f"group_cbv_mlp_k{K}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    train_ds = Sector14GroupCBVReconDataset(df, train_tics, t_range, "area", K,
                                            area_bases=bases, ridge_lambda=RIDGE_LAMBDA)
    val_ds = Sector14GroupCBVReconDataset(df, val_tics, t_range, "area", K,
                                          area_bases=bases, ridge_lambda=RIDGE_LAMBDA)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics
    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED), drop_last=True)

    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    load_frozen_teacher(model, teacher_sel_path)       # Stage-A EMA -> frozen teacher
    teacher_hash = model.teacher_hash()
    print(f"frozen Stage-A teacher hash {teacher_hash[:16]}...", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    fields = ["epoch", "train_loss", "val_loss", "val_camccd_bacc", "camera_acc",
              "ccd_acc", "effective_rank", "teacher_hash_ok"]
    metrics_path = os.path.join(ART_DIR, f"metrics_stageB_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    e0_camccd, e0_cam, e0_ccd, e0_erank = probe_b(model, train_ds, val_ds)
    print(f"[B] epoch 0 (init): camccd={e0_camccd:.4f} cam={e0_cam:.4f} "
          f"ccd={e0_ccd:.4f} erank={e0_erank:.1f}", flush=True)
    best = {"camccd": e0_camccd, "epoch": 0, "camera": e0_cam, "ccd": e0_ccd,
            "erank": e0_erank}
    torch.save(model.student.state_dict(), f"{ckpt_base}_best_student_encoder.pth")
    since_best = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_b(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch_b(model, val_loader)
        if model.teacher_hash() != teacher_hash:
            raise RuntimeError("FROZEN TEACHER CHANGED -- protocol violation")
        camccd, camera, ccd, erank = probe_b(model, train_ds, val_ds)
        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_camccd_bacc": camccd, "camera_acc": camera, "ccd_acc": ccd,
                 "effective_rank": erank, "teacher_hash_ok": True})
        marker = ""
        if camccd > best["camccd"]:
            best = {"camccd": camccd, "epoch": epoch, "camera": camera,
                    "ccd": ccd, "erank": erank}
            torch.save(model.state_dict(), f"{ckpt_base}_best.pth")
            torch.save(model.student.state_dict(), f"{ckpt_base}_best_student_encoder.pth")
            since_best = 0
            marker = " <- best"
        else:
            since_best += 1
        print(f"[B] epoch {epoch:02d}: train={train_loss:.5f} val={val_loss:.5f} "
              f"camccd={camccd:.4f} cam={camera:.4f} ccd={ccd:.4f} erank={erank:.1f} "
              f"teacher_ok=True{marker}", flush=True)
        if epoch >= MIN_EPOCHS and since_best >= PATIENCE:
            print(f"[B] early stop at epoch {epoch}", flush=True)
            break

    selection = {"tag": tag, "seed": SEED, "k": K, "ridge_lambda": RIDGE_LAMBDA,
                 "predictor_type": "mlp", "select_view": "online",
                 "metadata_guided": True, "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_student_encoder.pth",
                 "teacher_selection": teacher_sel_path, "teacher_hash": teacher_hash,
                 "teacher_hash_verified_every_epoch": True, "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    return best, (e0_camccd, e0_cam, e0_ccd, e0_erank), teacher_hash, train_ds, val_ds


# ------------------------------------------------------------- diagnostics
def diagnostic_figure(bases, val_ds):
    areas = list(val_ds.group_list)[:6]
    x = np.arange(val_ds.X.shape[1])
    fig, axes = plt.subplots(max(len(areas), 1), 4, figsize=(18, 3 * max(len(areas), 1)))
    axes = np.atleast_2d(axes)
    for row, area in enumerate(areas):
        grp = sorted(val_ds.group_rows[area])[:K]
        median, log_mad, valid, _ = group_statistics(
            val_ds.X[grp], val_ds.M[grp], val_ds.min_valid)
        recon = ridge_reconstruct(median, valid, bases[int(area)], RIDGE_LAMBDA)
        obs = valid > 0

        def masked(a):
            b = a.astype(float).copy()
            b[~obs] = np.nan
            return b

        axes[row, 0].plot(x, masked(median), lw=0.7, color="tab:blue")
        axes[row, 0].set_ylabel(f"area {int(area)}", fontsize=8)
        axes[row, 1].plot(x, masked(recon), lw=0.7, color="tab:orange")
        axes[row, 2].plot(x, masked(median - recon), lw=0.7, color="tab:green")
        axes[row, 3].plot(x, masked(log_mad), lw=0.7, color="0.4")
    for c, t in zip(range(4), ["8-star median", "K=8 reconstruction",
                               "median - reconstruction", "log-MAD"]):
        axes[0, c].set_title(t, fontsize=9)
    fig.suptitle(f"custom group CBV K={K} teacher targets (validation groups)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(ART_DIR, "group_cbv_teacher_targets.png"), dpi=130)
    plt.close(fig)

    rep = int(val_ds.group_list[0])
    B = bases[rep]
    fig2, ax = plt.subplots(figsize=(12, 6))
    for j in range(B.shape[1]):
        ax.plot(x, B[:, j], lw=0.7, label=f"CBV {j + 1}")
    ax.set_title(f"eight learned CBVs, area {rep}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(os.path.join(ART_DIR, "learned_cbvs_area.png"), dpi=130)
    plt.close(fig2)


def write_report(stage_a_best, stage_b_best, e0, teacher_hash, bases, train_ds, val_ds):
    torch.manual_seed(SEED + 1)
    rand = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                      n_layers=4, readout="mean",
                                      predictor_type="mlp").to(DEVICE).eval()
    rand_camccd, rand_cam, rand_ccd, _ = probe_b(rand, train_ds, val_ds)

    ref = {"mlp_jepa": None, "tx_jepa": None, "cbv_refined": None}
    source = "constants (no earlier results.json found)"
    rj = os.path.join(CBV_ART_DIR, "results.json")
    if os.path.exists(rj):
        with open(rj) as fh:
            r = json.load(fh).get("results", {})
        ref["mlp_jepa"] = r.get("mlp_jepa_encoder", {}).get("val_camccd_bacc")
        ref["tx_jepa"] = r.get("tx_jepa_encoder", {}).get("val_camccd_bacc")
        ref["cbv_refined"] = r.get("cbv_refined", {}).get("val_camccd_bacc")
        source = rj

    def diff(name, val):
        return (f"- group-CBV-MLP - {name}: {stage_b_best['camccd'] - val:+.4f}"
                if val is not None else f"- group-CBV-MLP - {name}: n/a (not in artifacts)")

    lines = [
        f"# Two-stage custom group-level TGLC CBV JEPA (K={K}, MLP) -- validation-only pilot",
        "", f"git commit: {git_commit()}",
        f"config: seed={SEED} K={K} ridge_lambda={RIDGE_LAMBDA} epochs<= {EPOCHS} "
        f"min={MIN_EPOCHS} patience={PATIENCE} predictor=mlp",
        f"CBV bases: {len(bases)} areas x exactly {K} uncentered group-median components", "",
        "## Stage A (regional teacher)",
        f"- best validation epoch: {stage_a_best['epoch']}",
        f"- val area balanced acc: {stage_a_best['area_bacc']:.4f}",
        f"- val camCCD balanced acc: {stage_a_best['camccd_bacc']:.4f}",
        f"- effective rank: {stage_a_best['erank']:.1f}", "",
        "## Stage B (frozen online encoder probe, validation)",
        f"- best validation epoch: {stage_b_best['epoch']}",
        f"- camCCD balanced acc: {stage_b_best['camccd']:.4f}",
        f"- camera acc: {stage_b_best['camera']:.4f}   CCD acc: {stage_b_best['ccd']:.4f}",
        f"- effective rank: {stage_b_best['erank']:.1f}",
        f"- epoch-0 (init) camCCD: {e0[0]:.4f}", "",
        "## Comparisons (identical probe harness)",
        f"- random S4D camCCD (matched rerun): {rand_camccd:.4f} "
        f"(cam {rand_cam:.4f}, ccd {rand_ccd:.4f})",
        f"- MLP-JEPA reference: {ref['mlp_jepa']}  [REFERENCE, not a matched rerun]",
        f"- Transformer-JEPA reference: {ref['tx_jepa']}  [REFERENCE, not a matched rerun]",
        f"- CBV-refined reference: {ref['cbv_refined']}  [REFERENCE, not a matched rerun]",
        f"- reference source: {source}", "",
        "## Score differences (this run - baseline)",
        diff("random", rand_camccd),
        diff("MLP-JEPA ref", ref["mlp_jepa"]),
        diff("tx-JEPA ref", ref["tx_jepa"]),
        diff("CBV-refined ref", ref["cbv_refined"]), "",
        "## Guarantees",
        f"- Stage A used 16 unique same-area training stars per example "
        f"(two disjoint {K}-star groups).",
        "- Stage B excluded the context star from its 8 teacher stars.",
        "- CBV bases used TRAIN TICs only; validation/test TICs never contributed.",
        "- The frozen regional teacher was verified bit-identical every Stage-B epoch.",
        "- The test split was never loaded or evaluated.",
        "- METADATA-GUIDED: area/region membership selects each star's basis.", "",
    ]
    with open(os.path.join(ART_DIR, "final_summary.md"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: two-stage group-CBV K={K} ridge={RIDGE_LAMBDA} mlp predictor "
          f"epochs<= {EPOCHS} min={MIN_EPOCHS} patience={PATIENCE} device={DEVICE}",
          flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    fit_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", K)
    assert not set(fit_ds.tics) & test_tics, "test TIC in basis-fit set"
    bases = build_or_load_area_bases(fit_ds.X, fit_ds.M, fit_ds.areas,
                                     sorted(fit_ds.tics), K, ART_DIR, K, GROUP_MIN_VALID)
    print(f"area CBV bases: {len(bases)} areas, exactly {K} uncentered "
          f"group-median components, TRAIN stars only", flush=True)

    print("=== STAGE A: regional teacher ===", flush=True)
    teacher_sel_path, stage_a_best = train_stage_a(
        df, train_tics, val_tics, test_tics, t_range, bases)

    print("=== STAGE B: single-star student vs frozen teacher ===", flush=True)
    stage_b_best, e0, teacher_hash, train_ds, val_ds = train_stage_b(
        df, train_tics, val_tics, test_tics, t_range, bases, teacher_sel_path)

    diagnostic_figure(bases, val_ds)
    write_report(stage_a_best, stage_b_best, e0, teacher_hash, bases, train_ds, val_ds)
    print(f"wrote {ART_DIR}/final_summary.md and diagnostic figures", flush=True)


if __name__ == "__main__":
    main()
