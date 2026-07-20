# All this code is from Claude
"""CBV-refinement pilot for the fixed-teacher Transformer-JEPA.

METADATA-GUIDED PRETRAINING, not label-free JEPA: each per-chip PCA/CBV basis
is defined by camera/CCD membership, so chip metadata enters the teacher
target. The single-star student and the frozen probe stay label-free; only
the teacher's group target is now CBV-filtered.

Change vs the Transformer-JEPA run: the 8 same-area target stars are each
projected onto their own chip's rank-64 PCA/CBV basis (masked least squares,
TRAIN-only bases, no interpolation) BEFORE the median / log-MAD fingerprint
is formed. Student, transformer predictor, frozen teacher, and JEPA loss are
unchanged; init is the current best Transformer-JEPA checkpoint.

    python -m src.instrument_v2.train_cbv_refinement_jepa
Env: SEED, EPOCHS(<=10), LR(3e-4), MIN_EPOCHS, PATIENCE, K_CBV(64),
     TX_CHECKPOINT, TEACHER_SELECTION, CBV_ART_DIR, CBV_CKPT_DIR,
     EPOCH0_REF(0.4559), EPOCH0_TOL(0.002), MAX_BATCHES, ...
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
    group_statistics,
)
from src.instrument_v2.diagnose_chip_common_signal import fit_chip_bases
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA,
    fixed_teacher_loss,
)
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import (
    effective_rank,
    git_commit,
    seed_worker,
)

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
K_CBV = int(os.environ.get("K_CBV", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "3e-4"))
VARW = float(os.environ.get("VARW", "0.5"))
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "4"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
BATCH = int(os.environ.get("BATCH", "64"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
EPOCH0_REF = float(os.environ.get("EPOCH0_REF", "0.4559"))
EPOCH0_TOL = float(os.environ.get("EPOCH0_TOL", "0.002"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ART_DIR = os.environ.get(
    "CBV_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "cbv_refinement_screen"))
CKPT_DIR = os.environ.get(
    "CBV_CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/cbv_refinement_screen")
TX_CHECKPOINT = os.environ.get(
    "TX_CHECKPOINT",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/transformer_encoder_screen/"
    "frtstudent_tx_k8_s0_best.pth")


# ------------------------------------------------------------- CBV bases
def train_tic_hash(tics):
    return hashlib.sha256("\n".join(sorted(map(str, tics)))
                          .encode()).hexdigest()[:16]


def build_or_load_bases(X, M, chips, tics, k_max, cache_dir):
    """Per-chip rank-k_max PCA/CBV bases from TRAIN rows only, cached by the
    train-TIC-set hash so val/test can never contaminate a basis."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"chip_bases_k{k_max}_{train_tic_hash(tics)}"
    npz = os.path.join(cache_dir, tag + ".npz")
    if os.path.exists(npz):
        data = np.load(npz, allow_pickle=True)
        return {int(c): (data[f"mean_{c}"], data[f"comp_{c}"], int(data[f"n_{c}"]))
                for c in data["chips"]}
    bases, skipped = fit_chip_bases(X, M, chips, k_max)
    store = {"chips": np.array(sorted(bases))}
    for chip, (mean, comp, n) in bases.items():
        store[f"mean_{chip}"] = mean
        store[f"comp_{chip}"] = comp
        store[f"n_{chip}"] = np.array(n)
    np.savez(npz, **store)
    with open(os.path.join(cache_dir, tag + ".json"), "w") as handle:
        json.dump({"train_tic_hash": train_tic_hash(tics), "k_max": k_max,
                   "n_train_tics": len(tics), "skipped_chips": skipped,
                   "git_commit": git_commit()}, handle, indent=2)
    return bases


def reconstruct_curve(x, m, mean, components, k):
    """Masked least-squares CBV projection of one star's systematic curve.

    Coefficients fit on OBSERVED cadences only (never fill/interpolate);
    the reconstruction is evaluated on all cadences and later masked back to
    the star's observed cadences by the caller."""
    obs = m > 0
    k = min(k, components.shape[0])
    if obs.sum() <= k + 1 or k == 0:
        return mean.astype(np.float32)
    A = components[:k][:, obs].T
    coef, *_ = np.linalg.lstsq(A, (x - mean)[obs].astype(np.float64), rcond=None)
    return (mean + components[:k].T @ coef).astype(np.float32)


class Sector14CBVGroupStatDataset(Sector14GroupStatDataset):
    """Same as Sector14GroupStatDataset, but the 8 target stars are CBV-
    projected onto their per-chip bases before median / log-MAD. The context
    star (student input) and the .X/.M used for frozen probing stay RAW."""

    def __init__(self, *args, bases=None, k_cbv=K_CBV, **kwargs):
        super().__init__(*args, **kwargs)
        if bases is None:
            raise ValueError("CBV dataset needs train-only per-chip bases")
        self.bases = bases
        self.k_cbv = k_cbv

    def _reconstruct_targets(self, targets):
        recon = np.empty((len(targets), self.X.shape[1]), dtype=np.float32)
        for i, row in enumerate(targets):
            chip = int(self.chips[row])
            if chip in self.bases:
                mean, comp, _ = self.bases[chip]
                recon[i] = reconstruct_curve(self.X[row], self.M[row],
                                             mean, comp, self.k_cbv)
            else:
                recon[i] = self.X[row]           # no basis -> raw fallback
        return recon

    def __getitem__(self, idx):
        context, targets, group = self._sample_item()
        recon = self._reconstruct_targets(targets)
        median, log_mad, valid, n_observed = group_statistics(
            recon, self.M[targets], self.min_valid)
        return (torch.tensor(self.X[context]), torch.tensor(self.M[context]),
                torch.tensor(median), torch.tensor(log_mad),
                torch.tensor(valid), torch.tensor(n_observed),
                torch.tensor(group, dtype=torch.int64))


def build_model():
    return FixedTeacherInstrumentJEPA(
        n_tokens=16, token_dim=16, d_model=256, n_layers=4,
        readout="mean", predictor_type="transformer")


def stack_stats(median, log_mad):
    return torch.stack([median, log_mad], dim=-1)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m, median, log_mad, valid, _, _ = \
                [t.to(DEVICE) for t in batch]
            prediction, target, tokens = model(
                ctx_f, ctx_m, stack_stats(median, log_mad), valid)
            loss = fixed_teacher_loss(prediction, target, tokens, valid,
                                      var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # NO EMA / NO teacher update: teacher is frozen.
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"cbv_refine_tx_k8_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)

    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: {tag} epochs={EPOCHS} lr={LR} k_cbv={K_CBV} "
          f"device={DEVICE}  (metadata-guided: chip defines each CBV basis)",
          flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    fit_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", K)
    assert not set(fit_ds.tics) & test_tics, "test TIC in basis-fitting set"
    bases = build_or_load_bases(fit_ds.X, fit_ds.M, fit_ds.chips,
                                sorted(fit_ds.tics), K_CBV, ART_DIR)
    print(f"CBV bases: {len(bases)} chips, rank {K_CBV}, TRAIN stars only",
          flush=True)

    train_ds = Sector14CBVGroupStatDataset(df, train_tics, t_range, "area", K,
                                           bases=bases, k_cbv=K_CBV)
    val_ds = Sector14CBVGroupStatDataset(df, val_tics, t_range, "area", K,
                                         bases=bases, k_cbv=K_CBV)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED)),
        "val": DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))}

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(TX_CHECKPOINT, map_location=DEVICE))
    teacher_hash = model.teacher_hash()
    print(f"init from {TX_CHECKPOINT}", flush=True)
    print(f"frozen teacher hash {teacher_hash[:16]}...", flush=True)

    def encoder_probe():
        tr = individual_latents(model, train_ds, "online")
        va = individual_latents(model, val_ds, "online")
        return (fast_probe(tr, train_ds.chips, va, val_ds.chips),
                effective_rank(va))

    # --- epoch-0 reproducibility gate ---
    epoch0_bacc, epoch0_erank = encoder_probe()
    print(f"epoch 0 (init) encoder camccd = {epoch0_bacc:.4f} "
          f"(ref {EPOCH0_REF:.4f}) erank {epoch0_erank:.1f}", flush=True)
    if abs(epoch0_bacc - EPOCH0_REF) > EPOCH0_TOL:
        raise RuntimeError(
            f"epoch-0 probe {epoch0_bacc:.4f} differs from reference "
            f"{EPOCH0_REF:.4f} by > {EPOCH0_TOL}; aborting (harness mismatch)")

    fields = ["epoch", "train_loss", "val_loss", "val_camccd_bacc",
              "effective_rank", "teacher_hash_ok"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"epoch": 0, "train_loss": "", "val_loss": "",
                         "val_camccd_bacc": epoch0_bacc,
                         "effective_rank": epoch0_erank, "teacher_hash_ok": True})

    best = {"camccd_bacc": epoch0_bacc, "epoch": 0,
            "effective_rank": epoch0_erank}
    torch.save(model.student.state_dict(), f"{ckpt_base}_best_student_encoder.pth")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)
    since_best = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], optimizer)
        val_loss = run_epoch(model, loaders["val"])
        if model.teacher_hash() != teacher_hash:
            raise RuntimeError("FROZEN TEACHER CHANGED -- protocol violation")
        bacc, erank = encoder_probe()

        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_camccd_bacc": bacc, "effective_rank": erank,
                 "teacher_hash_ok": True})
        marker = ""
        if bacc > best["camccd_bacc"]:
            best = {"camccd_bacc": bacc, "epoch": epoch, "effective_rank": erank}
            torch.save(model.state_dict(), f"{ckpt_base}_best.pth")
            torch.save(model.student.state_dict(),
                       f"{ckpt_base}_best_student_encoder.pth")
            since_best = 0
            marker = " <- best"
        else:
            since_best += 1
        print(f"epoch {epoch:02d}: train={train_loss:.5f} val={val_loss:.5f} "
              f"camccd={bacc:.4f} erank={erank:.1f}{marker}", flush=True)
        if epoch >= MIN_EPOCHS and since_best >= PATIENCE:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    selection = {"tag": tag, "seed": SEED, "k": K, "k_cbv": K_CBV,
                 "predictor_type": "transformer", "select_view": "online",
                 "metadata_guided": True,
                 "note": ("CBV bases are defined by camera/CCD membership -> "
                          "metadata-guided pretraining, not label-free JEPA"),
                 "epoch0_camccd": epoch0_bacc, "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_student_encoder.pth",
                 "init_checkpoint": TX_CHECKPOINT, "teacher_hash": teacher_hash,
                 "teacher_hash_verified_every_epoch": True,
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as handle:
        json.dump(selection, handle, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
