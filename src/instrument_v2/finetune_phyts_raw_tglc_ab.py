from __future__ import annotations
"""Matched SUPERVISED fine-tuning A/B: does instrument-cleaning help when the
physics encoder + a 7-class head are fine-tuned (not just a frozen probe)?

  ARM A  filtered raw TGLC -> physics JEPA encoder -> 7-class head
  ARM B  same filtered raw -> frozen instrument JEPA + frozen decoder -> subtract
         the decoded instrument ON THE NATIVE CADENCE GRID -> ONE physics
         preprocessing -> physics JEPA encoder -> 7-class head

Both arms fine-tune the physics encoder + head (identical init per seed). The
instrument encoder/predictor/teacher/decoder are NEVER updated. Data/split reuse
the .629->.645 raw-TGLC experiment (GaiaDR3+sector match, qclean filter).

  per-seed:   SEED=<n> python -m src.instrument_v2.finetune_phyts_raw_tglc_ab
  aggregate:  python -m src.instrument_v2.finetune_phyts_raw_tglc_ab --aggregate-only
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.sector14_dataset import grid_curve_shared
from src.instrument_v2.eval_phyts_instrument_ab import (
    DEVICE, GRID, physics_grid, matched_split,
)
from src.instrument_v2.eval_phyts_raw_tglc_ab import (
    PHYTS_PATH, TGLC_PATH, INST_CKPT, DECODER_CKPT, GRID_RANGE,
    match_phyts_tglc, quality_filter,
)

# the exact physics checkpoint behind raw=0.629 / cleaned=0.645 (env-overridable)
PHYS_CKPT = os.environ.get("JEPA_CKPT",
                           "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth")
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "phyts_raw_tglc_finetune"))
CACHE = os.path.join(OUT_DIR, "arms_cache.npz")
REF_RAW, REF_CLEANED = 0.629, 0.645
HEAD_EPOCHS = 20
FT_MAX_EPOCHS = 80
PATIENCE = 10
HEAD_LR, BACKBONE_LR = 1e-3, 1e-4
BATCH = 128


# ------------------------------------------------------- native-grid cleaning
def decode_native_template(ft, ff, inst, decoder, t0, t1):
    """Decode the instrument template ONCE and return every quantity the native-
    and grid-cleaning arms need (X, mask, decoded template, median/scale, grid
    times). This is the exact front half of cleaned_native_flux -- unchanged
    numerics; it just lets a caller reuse one decode across several arms.
    `decoded` is None when the curve has <8 shared-grid bins (no cleaning)."""
    normed, med, mad = normalize_median_mad(ff)
    scale = 1.4826 * mad if 1.4826 * mad > 0 else 1.0
    X, M = grid_curve_shared(ft, normed, t0, t1, GRID)
    valid = M > 0
    grid_times = t0 + (np.arange(GRID) + 0.5) / GRID * (t1 - t0)
    decoded = None
    if valid.sum() >= 8:
        with torch.no_grad():
            z = inst.encode(torch.tensor(X, dtype=torch.float32, device=DEVICE)[None],
                            torch.tensor(M, dtype=torch.float32, device=DEVICE)[None], view="predicted")
            decoded = decoder(z).squeeze(0).detach().cpu().numpy()   # (1024,) normalized instrument
    return {"X": X, "M": M, "valid": valid, "decoded": decoded,
            "med": med, "scale": scale, "grid_times": grid_times}


def cleaned_native_flux(ft, ff, inst, decoder, t0, t1, template=None):
    """Subtract the decoded instrument ON the original cadence grid (no double
    resample): decode on the S14 shared grid, interpolate the template back to
    native times, subtract in flux units. Behaviour is unchanged; `template`
    lets a caller pass a precomputed decode (decode once, reuse)."""
    tpl = template if template is not None else decode_native_template(ft, ff, inst, decoder, t0, t1)
    if tpl["decoded"] is None:
        return np.asarray(ff, dtype=np.float64)               # too few bins -> no cleaning
    valid = tpl["valid"]
    dec_native = np.interp(np.asarray(ft, float), tpl["grid_times"][valid], tpl["decoded"][valid])
    return np.asarray(ff, dtype=np.float64) - dec_native * tpl["scale"]   # native-grid subtraction


def _ordered_hash(*cols):
    h = hashlib.sha256()
    for row in zip(*cols):
        h.update(("|".join(str(v) for v in row) + "\n").encode())
    return h.hexdigest()


# ------------------------------------------------------------- data preparation
def prepare_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next((c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id")
                     if c in phyts.columns), None)
    if gaia_col is None:
        raise RuntimeError("PhyTS has no Gaia id column")
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})

    tglc_cols = set(pq.read_schema(TGLC_PATH).names)
    flux_col = "aperture_flux" if "aperture_flux" in tglc_cols else "flux"
    tglc = pd.read_parquet(TGLC_PATH, columns=["TIC", "sector", "GAIADR3", "time", flux_col,
                                               "TESS_flags", "TGLC_flags"])
    tglc["TIC"] = tglc["TIC"].astype(str)
    tglc = tglc.rename(columns={flux_col: "aperture_flux"})
    matched, _ = match_phyts_tglc(phyts, tglc)
    n = len(matched)

    tics = matched["TIC"].to_numpy().astype(str)
    sectors = matched["sector"].to_numpy()
    gaia = matched["GAIADR3"].to_numpy()
    labels = np.array([CLASS_TO_IDX[l] for l in matched["label"]], dtype=np.int64)
    present = np.unique(labels)
    if len(present) != 7:
        raise RuntimeError(f"expected 7 classes, got {len(present)}")
    remap = {int(c): i for i, c in enumerate(present)}
    y = np.array([remap[int(v)] for v in labels], dtype=np.int64)      # 0..6 for the head

    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                      readout="mean", predictor_type="mlp").to(DEVICE)
    inst.load_state_dict(torch.load(INST_CKPT, map_location=DEVICE)); inst.eval()
    for p in inst.parameters():
        p.requires_grad_(False)
    decoder = build_decoder(1024).to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE)); decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    inst_hashes = np.array([state_hash(inst.teacher), state_hash(inst.student),
                            state_hash(inst.predictor), state_hash(decoder)])

    A_X = np.zeros((n, GRID), np.float32); A_M = np.zeros((n, GRID), np.float32)
    B_X = np.zeros((n, GRID), np.float32); B_M = np.zeros((n, GRID), np.float32)
    for i in range(n):
        ft, ff = quality_filter(matched["time"].iloc[i], matched["aperture_flux"].iloc[i],
                                matched["TESS_flags"].iloc[i], matched["TGLC_flags"].iloc[i])
        A_X[i], A_M[i] = physics_grid(ft, ff)
        B_X[i], B_M[i] = physics_grid(ft, cleaned_native_flux(ft, ff, inst, decoder, t0, t1))
        if i % 300 == 0:
            print(f"  building arms {i}/{n}", flush=True)
    inst_frozen_ok = np.array_equal(                        # real before/after check of the frozen models
        inst_hashes, [state_hash(inst.teacher), state_hash(inst.student),
                      state_hash(inst.predictor), state_hash(decoder)])
    if not inst_frozen_ok:
        raise RuntimeError("instrument/decoder hashes changed during arm building")

    train_full, test_idx = matched_split(tics, y)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sub_tr, sub_val = next(gss.split(train_full, y[train_full], groups=tics[train_full]))
    train_idx, val_idx = train_full[sub_tr], train_full[sub_val]

    out = dict(
        A_X=A_X, A_M=A_M, B_X=B_X, B_M=B_M, y=y, tics=tics, gaia=gaia, sectors=sectors,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, present=present,
        inst_frozen_ok=np.array(bool(inst_frozen_ok)),
        gaia_sector_sha256=np.array(_ordered_hash(gaia, sectors)),
        split_sha256=np.array(_ordered_hash(np.concatenate([train_idx, val_idx, test_idx]))))
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = f"{CACHE}.{os.getpid()}.tmp.npz"                  # atomic write: concurrent array tasks
    np.savez(tmp, **out)                                   # (.npz suffix so savez won't rename it)
    os.replace(tmp, CACHE)                                 # must never read a half-written cache
    return out


# --------------------------------------------------------------------- model
class FTModel(nn.Module):
    def __init__(self, backbone, feat_dim, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, flux, mask):
        z = self.backbone(flux.unsqueeze(-1), mask)
        return self.head(z.reshape(z.shape[0], -1))


def load_physics_backbone():
    m = build_latent_jepa()
    missing, unexpected = m.load_state_dict(torch.load(PHYS_CKPT, map_location="cpu"), strict=False)
    if missing or unexpected:                                  # strict: abort on any mismatch
        raise RuntimeError(f"physics ckpt key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    bb = m.target_encoder
    for p in bb.parameters():
        p.requires_grad_(True)
    return bb


# ------------------------------------------------------------------- training
def _predict(model, X, M, idx, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            out.append(model(X[j], M[j]).argmax(1).cpu())
    return torch.cat(out).numpy()


def _train_epoch(model, opt, ce, X, M, y, idx):
    model.train()
    perm = idx[torch.randperm(len(idx), device=idx.device)]
    for s in range(0, len(perm), BATCH):
        j = perm[s:s + BATCH]
        opt.zero_grad()
        loss = ce(model(X[j], M[j]), y[j])
        loss.backward()
        opt.step()


def train_arm(init_model, X, M, y, train_idx, val_idx, class_w, seed):
    torch.manual_seed(seed)
    model = copy.deepcopy(init_model).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=class_w.to(DEVICE))
    best_val, best_state = -1.0, None

    def maybe_best():
        nonlocal best_val, best_state
        vb = balanced_accuracy_score(y[val_idx].cpu().numpy(), _predict(model, X, M, val_idx))
        if vb > best_val:
            best_val = vb
            best_state = copy.deepcopy(model.state_dict())
        return vb

    for p in model.backbone.parameters():                      # phase 1: head only
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR)
    for _ in range(HEAD_EPOCHS):
        _train_epoch(model, opt, ce, X, M, y, train_idx)
        maybe_best()

    for p in model.backbone.parameters():                      # phase 2: fine-tune all
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": HEAD_LR},
                             {"params": model.backbone.parameters(), "lr": BACKBONE_LR}])
    since = 0
    for _ in range(FT_MAX_EPOCHS):
        _train_epoch(model, opt, ce, X, M, y, train_idx)
        prev = best_val
        maybe_best()
        since = 0 if best_val > prev else since + 1
        if since >= PATIENCE:
            break

    model.load_state_dict(best_state)                          # best-val checkpoint
    return model, float(best_val)


def eval_test(model, X, M, y, test_idx, present):
    pred = _predict(model, X, M, test_idx)
    yte = y[test_idx].cpu().numpy()
    bacc = float(balanced_accuracy_score(yte, pred))
    rec = recall_score(yte, pred, labels=list(range(len(present))), average=None, zero_division=0)
    return bacc, {CLASSES[int(present[i])]: float(r) for i, r in enumerate(rec)}


def run_seed(seed):
    d = prepare_data()
    present = d["present"]
    X = {a: torch.tensor(d[f"{a}_X"], device=DEVICE) for a in ("A", "B")}
    M = {a: torch.tensor(d[f"{a}_M"], device=DEVICE) for a in ("A", "B")}
    y = torch.tensor(d["y"], device=DEVICE)
    train_idx = torch.tensor(d["train_idx"], device=DEVICE)
    val_idx = torch.tensor(d["val_idx"], device=DEVICE)
    test_idx = torch.tensor(d["test_idx"], device=DEVICE)

    counts = np.bincount(d["y"][d["train_idx"]], minlength=len(present))
    class_w = torch.tensor(counts.sum() / (len(present) * np.maximum(counts, 1)), dtype=torch.float32)

    bb = load_physics_backbone().eval()
    with torch.no_grad():                                  # size the head (RNG-free: before the seed)
        feat_dim = bb(torch.zeros(1, GRID, 1), torch.ones(1, GRID)).reshape(1, -1).shape[1]
    torch.manual_seed(seed)                                # seed only the head init
    base = FTModel(load_physics_backbone(), feat_dim, len(present))
    model_A_init, model_B_init = copy.deepcopy(base), copy.deepcopy(base)
    sa, sb = model_A_init.state_dict(), model_B_init.state_dict()
    started_identical = all(torch.equal(sa[k], sb[k]) for k in sa)
    assert started_identical, "arms did not start bit-identical"

    m_a, val_a = train_arm(model_A_init, X["A"], M["A"], y, train_idx, val_idx, class_w, seed)
    test_a, rec_a = eval_test(m_a, X["A"], M["A"], y, test_idx, present)
    m_b, val_b = train_arm(model_B_init, X["B"], M["B"], y, train_idx, val_idx, class_w, seed)
    test_b, rec_b = eval_test(m_b, X["B"], M["B"], y, test_idx, present)

    result = {
        "seed": int(seed),
        "raw": {"val_bacc": val_a, "test_bacc": test_a, "test_recall": rec_a},
        "cleaned": {"val_bacc": val_b, "test_bacc": test_b, "test_recall": rec_b},
        "cleaned_minus_raw_test": test_b - test_a,
        "arms_started_identical": bool(started_identical),
        "instrument_decoder_hashes_unchanged": bool(d["inst_frozen_ok"]),  # verified in prepare_data
        "test_evaluated_after_val_selection": True,
    }
    path = os.path.join(OUT_DIR, f"seed_{int(seed)}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"seed {seed}: raw_test={test_a:.4f} cleaned_test={test_b:.4f} "
          f"diff={test_b - test_a:+.4f}", flush=True)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def aggregate():
    seeds = []
    for s in (0, 1, 2):
        p = os.path.join(OUT_DIR, f"seed_{s}.json")
        if os.path.exists(p):
            seeds.append(json.load(open(p)))
    if not seeds:
        raise SystemExit("no per-seed results found")
    raw_test = np.array([r["raw"]["test_bacc"] for r in seeds])
    clean_test = np.array([r["cleaned"]["test_bacc"] for r in seeds])
    diff = np.array([r["cleaned_minus_raw_test"] for r in seeds])
    d = prepare_data()

    summary = {
        "phys_ckpt": PHYS_CKPT, "inst_ckpt": INST_CKPT, "decoder_ckpt": DECODER_CKPT,
        "git_commit": _git_commit(),
        "frozen_reference": {"raw": REF_RAW, "cleaned": REF_CLEANED},
        "per_seed": {r["seed"]: {
            "raw_val": r["raw"]["val_bacc"], "raw_test": r["raw"]["test_bacc"],
            "cleaned_val": r["cleaned"]["val_bacc"], "cleaned_test": r["cleaned"]["test_bacc"],
            "cleaned_minus_raw_test": r["cleaned_minus_raw_test"],
            "raw_test_recall": r["raw"]["test_recall"],
            "cleaned_test_recall": r["cleaned"]["test_recall"]} for r in seeds},
        "raw_test_mean": float(raw_test.mean()), "raw_test_std": float(raw_test.std()),
        "cleaned_test_mean": float(clean_test.mean()), "cleaned_test_std": float(clean_test.std()),
        "cleaned_minus_raw_mean": float(diff.mean()), "cleaned_minus_raw_std": float(diff.std()),
        "n_seeds": len(seeds),
        "gaia_sector_sha256": str(d["gaia_sector_sha256"]),
        "split_sha256": str(d["split_sha256"]),
        "arms_started_identical": all(r["arms_started_identical"] for r in seeds),
        "instrument_decoder_hashes_unchanged": all(
            r["instrument_decoder_hashes_unchanged"] for r in seeds),
        "test_evaluated_after_val_selection": all(
            r["test_evaluated_after_val_selection"] for r in seeds),
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"raw {raw_test.mean():.4f}+-{raw_test.std():.4f}  "
          f"cleaned {clean_test.mean():.4f}+-{clean_test.std():.4f}  "
          f"diff {diff.mean():+.4f}+-{diff.std():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.aggregate_only:
        aggregate()
    else:
        seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SEED", "0")))
        run_seed(seed)


if __name__ == "__main__":
    main()
