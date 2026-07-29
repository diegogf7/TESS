from __future__ import annotations
"""Matched A/B diagnostic: does subtracting the decoded instrument template help
or hurt PhyTS classification with the frozen physics JEPA?

  ARM A  original PhyTS curve -> physics normalize/resample -> physics JEPA latent
  ARM B  original curve -> instrument JEPA + direct decoder -> subtract decoded
         instrument (no scale fit) -> back to flux -> physics normalize/resample
         -> physics JEPA latent

Same rows/order/TICs/labels/split in both arms; only the input curve differs.
Nothing is trained or modified. KNN(20) probe, matching the physics evaluation.

  python -m src.instrument_v2.eval_phyts_instrument_ab
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.data.data import (
    CLASSES,
    CLASS_TO_IDX,
    DualEvalDataset,
    normalize,
    resample_to_grid,
)
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.sector14_dataset import grid_curve_shared

GRID = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet")
PHYS_CKPT = os.environ.get(
    "JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_transformer.pth")
INST_CKPT = os.environ.get(
    "INST_CKPT",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1/"
    "group_cbv_mlp_g32_r8_mv16_s0_best.pth")
DECODER_CKPT = os.environ.get(
    "DECODER_CKPT",
    "artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/single_star_decode/decoder.pth")
GRID_RANGE = os.environ.get(
    "GRID_RANGE", "artifacts/instrument_v2/sector14_jepa_dense_v2/grid_range.json")
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "phyts_instrument_ab"))


# --------------------------------------------------------------- arm builders
def physics_grid(time, flux):
    """Exact DualEvalDataset preprocessing: physics normalize + 1024 resample."""
    nf = normalize(np.asarray(flux, dtype=np.float64))
    return resample_to_grid(np.asarray(time, dtype=np.float64), nf, GRID)


def instrument_cleaned_curve(time, flux, model_inst, decoder, t0, t1, basis=None, area_vec=None):
    """Instrument-clean a curve: median/MAD-normalize good cadences onto the S14
    shared grid, decode the instrument template, subtract it (NO scale fit), and
    return the cleaned curve in flux units at its observed grid times.

    basis is None (direct): `decoder` is the 1024-output decoder; template =
    decoder(z). basis given (cbv): `decoder` is build_decoder(8); template =
    basis @ 8-weights. area_vec given (weights_area_conditioned): the area one-hot
    is concatenated onto the latent before the decoder. Everything after the
    template is identical."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(time) & np.isfinite(flux)          # finite only; no TESS/TGLC flags
    tg, fg = time[good], flux[good]
    normed, med, mad = normalize_median_mad(fg)           # instrument median/MAD
    scale = 1.4826 * mad if 1.4826 * mad > 0 else 1.0
    X, M = grid_curve_shared(tg, normed, t0, t1, GRID)    # instrument input + observed mask
    valid = M > 0
    if valid.sum() < 8:                                   # curve misaligned to the S14 grid
        return tg, fg                                     # fall back to the raw good curve
    with torch.no_grad():
        z = model_inst.encode(
            torch.tensor(X, dtype=torch.float32, device=DEVICE)[None],
            torch.tensor(M, dtype=torch.float32, device=DEVICE)[None], view="predicted")
        if area_vec is not None:                          # concat(latent_256, area one-hot)
            z = torch.cat([z.reshape(1, -1),
                           torch.tensor(area_vec, dtype=torch.float32, device=DEVICE)[None]], dim=1)
        out = decoder(z).squeeze(0).detach().cpu().numpy()          # (1024,) direct or (8,) cbv
    decoded = out if basis is None else (np.asarray(basis) @ out)   # cbv: (1024,8)@(8,) = (1024,)
    cleaned_norm = X - decoded                            # subtract decoded instrument (no fit)
    cleaned_flux = cleaned_norm * scale + med             # back to flux units (orig median/scale)
    grid_times = t0 + (np.arange(GRID) + 0.5) / GRID * (t1 - t0)
    return grid_times[valid], cleaned_flux[valid]


def encode_physics(model, X, M, batch=256):
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            f = torch.tensor(X[s:s + batch], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(M[s:s + batch], dtype=torch.float32, device=DEVICE)
            z = model.encode(f, m)
            outs.append(z.reshape(z.shape[0], -1).detach().cpu().numpy())
    return np.concatenate(outs)


# ------------------------------------------------------------- classification
def matched_split(tics, y):
    """One deterministic TIC-disjoint train/test split, shared by both arms."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    return next(splitter.split(np.arange(len(tics)), y, groups=tics))


def assert_classes_present(y, train_idx, test_idx):
    present = set(np.unique(y).tolist())
    if not (present <= set(y[train_idx].tolist()) and present <= set(y[test_idx].tolist())):
        raise RuntimeError("a class is absent from train or test -- aborting")


def classify(latents, y, train_idx, test_idx, present):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(latents[train_idx])       # fit on train latents only
    x_te = scaler.transform(latents[test_idx])
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(x_tr, y[train_idx])
    pred = knn.predict(x_te)
    acc = float(balanced_accuracy_score(y[test_idx], pred))
    rec = recall_score(y[test_idx], pred, labels=present, average=None, zero_division=0)
    return acc, {CLASSES[c]: float(r) for c, r in zip(present, rec)}


def ordered_hash(tics, y):
    h = hashlib.sha256()
    for t, lab in zip(tics, y):
        h.update(f"{t}|{int(lab)}\n".encode())
    return h.hexdigest()


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    df = pd.read_parquet(DATA_PATH)
    df = df[df["sector"] == 14].reset_index(drop=True)
    n = len(df)
    tics = df["TIC"].to_numpy().astype(str)
    y = np.array([CLASS_TO_IDX[l] for l in df["label"]], dtype=np.int64)
    present = np.unique(y)
    print(f"PhyTS sector-14 curves: {n} | classes: {[CLASSES[c] for c in present]}", flush=True)
    probe = range(min(n, 50))
    tmin = min(float(np.nanmin(np.asarray(df["time"].iloc[i], float))) for i in probe)
    tmax = max(float(np.nanmax(np.asarray(df["time"].iloc[i], float))) for i in probe)
    print(f"PhyTS time ~[{tmin:.1f}, {tmax:.1f}] vs S14 grid [{t0:.1f}, {t1:.1f}] "
          f"(must overlap or Arm B degenerates to Arm A)", flush=True)

    # --- frozen models (loaded before any inference) -------------------------
    phys = build_latent_jepa().to(DEVICE)
    phys.load_state_dict(torch.load(PHYS_CKPT, map_location=DEVICE), strict=False)
    _freeze(phys)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                      readout="mean", predictor_type="mlp").to(DEVICE)
    inst.load_state_dict(torch.load(INST_CKPT, map_location=DEVICE))
    _freeze(inst)
    decoder = _freeze(build_decoder(1024).to(DEVICE))
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    _freeze(decoder)
    hashes = {"phys": state_hash(phys), "inst_teacher": state_hash(inst.teacher),
              "inst_student": state_hash(inst.student),
              "inst_predictor": state_hash(inst.predictor), "decoder": state_hash(decoder)}

    # --- build both arms' physics-grid inputs on identical rows --------------
    A_X = np.zeros((n, GRID), np.float32); A_M = np.zeros((n, GRID), np.float32)
    B_X = np.zeros((n, GRID), np.float32); B_M = np.zeros((n, GRID), np.float32)
    for i in range(n):
        t = df["time"].iloc[i]; f = df["flux"].iloc[i]
        A_X[i], A_M[i] = physics_grid(t, f)
        ct, cf = instrument_cleaned_curve(t, f, inst, decoder, t0, t1)
        B_X[i], B_M[i] = physics_grid(ct, cf)
        if i % 200 == 0:
            print(f"  building arms {i}/{n}", flush=True)

    # sentinel: arm A reproduces the existing DualEvalDataset for the first 32 rows
    ref = DualEvalDataset(DATA_PATH, GRID)
    ref.df = df
    for i in range(min(32, n)):
        gf, ob, _, _ = ref[i]
        assert np.allclose(A_X[i], gf.numpy(), atol=1e-5), f"arm-A flux != DualEvalDataset row {i}"
        assert np.allclose(A_M[i], ob.numpy()), f"arm-A mask != DualEvalDataset row {i}"

    lat_a = encode_physics(phys, A_X, A_M)
    lat_b = encode_physics(phys, B_X, B_M)
    assert np.isfinite(lat_a).all() and np.isfinite(lat_b).all(), "non-finite latents"
    assert lat_a.shape[0] == lat_b.shape[0] == n

    # frozen: nothing changed
    assert state_hash(phys) == hashes["phys"], "physics JEPA changed"
    assert state_hash(inst.teacher) == hashes["inst_teacher"], "instrument teacher changed"
    assert state_hash(inst.student) == hashes["inst_student"], "instrument student changed"
    assert state_hash(inst.predictor) == hashes["inst_predictor"], "instrument predictor changed"
    assert state_hash(decoder) == hashes["decoder"], "decoder changed"

    # --- one matched split, both arms, KNN probe ----------------------------
    train_idx, test_idx = matched_split(tics, y)
    assert_classes_present(y, train_idx, test_idx)
    raw_acc, raw_recall = classify(lat_a, y, train_idx, test_idx, present)
    clean_acc, clean_recall = classify(lat_b, y, train_idx, test_idx, present)

    report = {
        "n_curves": int(n),
        "n_classes": int(len(present)),
        "classes": [CLASSES[c] for c in present],
        "raw_balanced_accuracy": raw_acc,
        "instrument_cleaned_balanced_accuracy": clean_acc,
        "cleaned_minus_raw": clean_acc - raw_acc,
        "raw_per_class_recall": raw_recall,
        "cleaned_per_class_recall": clean_recall,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "tic_row_sha256": ordered_hash(tics, y),
        "phys_ckpt": PHYS_CKPT, "inst_ckpt": INST_CKPT, "decoder_ckpt": DECODER_CKPT,
        "model_hashes_unchanged": True, "phyts_test_split_loaded": False,
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"raw={raw_acc:.6f}  cleaned={clean_acc:.6f}  diff={clean_acc - raw_acc:+.6f}",
          flush=True)
    print(f"wrote {OUT_DIR}/final_summary.json", flush=True)


if __name__ == "__main__":
    main()
