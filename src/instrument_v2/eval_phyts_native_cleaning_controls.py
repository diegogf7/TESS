from __future__ import annotations
"""Decisive frozen 2x2 preprocessing diagnostic. Did the frozen 0.629 -> 0.645
"cleaning helps" come from ACTUAL instrument subtraction, or from the extra
shared-grid resampling/smoothing baked into the old cleaned arm?

Four matched arms, one frozen physics JEPA, one KNN(20) probe, one TIC-disjoint
split, all built from the SAME quality-filtered native (time, flux) and ONE
instrument decode per curve:

  A  native raw      physics preprocess(native)                         -> ref (~0.629)
  B  native cleaned  subtract decoded template at native times, then    -> corrected cleaning
                     physics preprocess ONCE   (== cleaned_native_flux)
  C  grid raw        raw on the shared grid, NO subtraction, back to     -> resampling ONLY
                     flux, physics preprocess  (old pipeline minus subtract)
  D  grid cleaned    raw on the shared grid, minus decoded, back to      -> old cleaned (~0.645)
                     flux, physics preprocess  (== instrument_cleaned_curve)

Decision:  B-A  = true native subtraction effect
           C-A  = resampling/smoothing artifact
           D-C  = subtraction under the old gridded pipeline
           D-A  = previous combined 0.629->0.645

  python -m src.instrument_v2.eval_phyts_native_cleaning_controls
"""

import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.eval_phyts_instrument_ab import (
    DEVICE, GRID, physics_grid, instrument_cleaned_curve, encode_physics,
    matched_split, assert_classes_present,
)
from src.instrument_v2.eval_phyts_raw_tglc_ab import (
    PHYTS_PATH, TGLC_PATH, INST_CKPT, DECODER_CKPT, GRID_RANGE,
    match_phyts_tglc, quality_filter, ordered_hash_tic_sector,
)
from src.instrument_v2.finetune_phyts_raw_tglc_ab import (
    decode_native_template, cleaned_native_flux,
)

PHYS_CKPT = os.environ.get("JEPA_CKPT",
                           "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth")
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "phyts_native_cleaning_controls"))
EXPECTED_TIC_SECTOR_SHA = "6a1796a05d4313479a39cf8fdf5e8b273544558c28692fdbba9df63dc8f425cc"
EXPECTED_N_MATCHED = 2409
N_BOOT = 2000

ARM_DESC = {
    "A": "native raw: quality-filtered native flux -> physics normalize/resample -> physics JEPA",
    "B": "native cleaned: subtract decoded instrument at native cadences (cleaned_native_flux), "
         "then physics normalize/resample ONCE -> physics JEPA",
    "C": "grid raw control: raw normalized onto shared grid, NO subtraction, back to flux at valid "
         "grid times -> physics normalize/resample -> physics JEPA (resampling only)",
    "D": "grid cleaned (old pipeline): raw onto shared grid MINUS decoded instrument, back to flux "
         "at valid grid times (instrument_cleaned_curve) -> physics normalize/resample -> physics JEPA",
}


def _sha(*cols):
    h = hashlib.sha256()
    for row in zip(*cols):
        h.update(("|".join(str(v) for v in row) + "\n").encode())
    return h.hexdigest()


def strict_load(model, path):
    """Load a checkpoint and abort on ANY missing/unexpected key."""
    missing, unexpected = model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint {path} key mismatch: "
                           f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def grid_curve_from_template(tpl, ft, ff, subtract):
    """Old gridded pipeline from a shared decode. subtract=False -> Arm C (raw on
    grid, no subtraction); subtract=True -> Arm D (== instrument_cleaned_curve).
    Falls back to the native raw curve when the curve has <8 shared-grid bins,
    matching instrument_cleaned_curve's fallback exactly."""
    if tpl["decoded"] is None:
        return np.asarray(ft, np.float64), np.asarray(ff, np.float64)
    valid = tpl["valid"]
    sub = tpl["decoded"] if subtract else 0.0
    cleaned_flux = (tpl["X"] - sub) * tpl["scale"] + tpl["med"]   # back to flux units (no fit)
    return tpl["grid_times"][valid], cleaned_flux[valid]


def classify_arm(latents, y, train_idx, test_idx, present):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(latents[train_idx])       # fit on TRAIN latents only
    knn = KNeighborsClassifier(n_neighbors=20).fit(x_tr, y[train_idx])
    pred = knn.predict(scaler.transform(latents[test_idx]))
    acc = float(balanced_accuracy_score(y[test_idx], pred))
    rec = recall_score(y[test_idx], pred, labels=present, average=None, zero_division=0)
    return acc, {CLASSES[c]: float(r) for c, r in zip(present, rec)}, pred


def paired_bootstrap(y_test, preds, present, n_boot=N_BOOT, seed=0):
    """Deterministic paired, class-stratified bootstrap: resample within each
    class (fixed class sizes), score every arm on the SAME resample, report 95%
    CIs for the four differences."""
    rng = np.random.default_rng(seed)
    class_idx = {c: np.where(y_test == c)[0] for c in present}
    acc = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(class_idx[c], len(class_idx[c]), replace=True)
                              for c in present])
        yb = y_test[idx]
        for k in preds:
            acc[k][b] = balanced_accuracy_score(yb, preds[k][idx])

    def ci(diff):
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return {"mean": float(diff.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}

    return {"B_minus_A": ci(acc["B"] - acc["A"]), "C_minus_A": ci(acc["C"] - acc["A"]),
            "D_minus_C": ci(acc["D"] - acc["C"]), "D_minus_A": ci(acc["D"] - acc["A"])}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    # --- PhyTS labels/metadata ONLY, matched to raw TGLC by GaiaDR3+sector ----
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
    y = np.array([CLASS_TO_IDX[l] for l in matched["label"]], dtype=np.int64)
    present = np.unique(y)

    tic_sector_sha = ordered_hash_tic_sector(tics, sectors)
    if n != EXPECTED_N_MATCHED:
        raise RuntimeError(f"n_matched={n} != {EXPECTED_N_MATCHED}")
    if tic_sector_sha != EXPECTED_TIC_SECTOR_SHA:
        raise RuntimeError(f"TIC/sector hash {tic_sector_sha} != prior frozen experiment")
    if len(present) != 7:
        raise RuntimeError(f"expected 7 classes, got {len(present)}")
    print(f"matched {n}/{EXPECTED_N_MATCHED} | tic_sector_sha OK | classes {len(present)}", flush=True)

    # --- frozen models, strict load ------------------------------------------
    phys = strict_load(build_latent_jepa().to(DEVICE), PHYS_CKPT)
    inst = strict_load(FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                                  readout="mean", predictor_type="mlp").to(DEVICE), INST_CKPT)
    decoder = strict_load(build_decoder(1024).to(DEVICE), DECODER_CKPT)
    hashes = {"phys": state_hash(phys), "inst_teacher": state_hash(inst.teacher),
              "inst_student": state_hash(inst.student),
              "inst_predictor": state_hash(inst.predictor), "decoder": state_hash(decoder)}

    # --- four arms, one decode per curve -------------------------------------
    X = {a: np.zeros((n, GRID), np.float32) for a in "ABCD"}
    M = {a: np.zeros((n, GRID), np.float32) for a in "ABCD"}
    for i in range(n):
        ft, ff = quality_filter(matched["time"].iloc[i], matched["aperture_flux"].iloc[i],
                                matched["TESS_flags"].iloc[i], matched["TGLC_flags"].iloc[i])
        tpl = decode_native_template(ft, ff, inst, decoder, t0, t1)     # ONE decode, reused below
        X["A"][i], M["A"][i] = physics_grid(ft, ff)
        X["B"][i], M["B"][i] = physics_grid(ft, cleaned_native_flux(ft, ff, inst, decoder, t0, t1, template=tpl))
        X["C"][i], M["C"][i] = physics_grid(*grid_curve_from_template(tpl, ft, ff, subtract=False))
        X["D"][i], M["D"][i] = physics_grid(*grid_curve_from_template(tpl, ft, ff, subtract=True))
        if i % 300 == 0:
            print(f"  building arms {i}/{n}", flush=True)

    lat = {a: encode_physics(phys, X[a], M[a]) for a in "ABCD"}
    for a in "ABCD":
        assert np.isfinite(lat[a]).all(), f"non-finite latents arm {a}"

    hashes_after = {"phys": state_hash(phys), "inst_teacher": state_hash(inst.teacher),
                    "inst_student": state_hash(inst.student),
                    "inst_predictor": state_hash(inst.predictor), "decoder": state_hash(decoder)}
    model_hashes_unchanged = hashes == hashes_after
    if not model_hashes_unchanged:
        raise RuntimeError("a frozen model changed during inference")

    # --- one shared split; KNN probe per arm ---------------------------------
    train_idx, test_idx = matched_split(tics, y)
    assert_classes_present(y, train_idx, test_idx)
    acc, recall, pred = {}, {}, {}
    for a in "ABCD":
        acc[a], recall[a], pred[a] = classify_arm(lat[a], y, train_idx, test_idx, present)

    y_test = y[test_idx]
    boot = paired_bootstrap(y_test, pred, present)

    # --- per-curve predictions (all four arms) -------------------------------
    split = np.array(["train"] * n, dtype=object)
    split[test_idx] = "test"
    test_pos = {int(gi): j for j, gi in enumerate(test_idx)}
    rows = []
    for gi in range(n):
        r = {"TIC": tics[gi], "GaiaDR3": gaia[gi], "true_label": CLASSES[y[gi]], "split": split[gi]}
        for a in "ABCD":
            r[f"pred_{a}"] = CLASSES[pred[a][test_pos[gi]]] if gi in test_pos else ""
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_curve_predictions.csv"), index=False)

    report = {
        "n_matched": int(n), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "classes": [CLASSES[c] for c in present],
        "balanced_accuracy": {a: acc[a] for a in "ABCD"},
        "per_class_recall": {a: recall[a] for a in "ABCD"},
        "differences": {
            "B_minus_A": acc["B"] - acc["A"], "C_minus_A": acc["C"] - acc["A"],
            "D_minus_C": acc["D"] - acc["C"], "D_minus_A": acc["D"] - acc["A"]},
        "bootstrap_ci95": boot, "n_bootstrap": N_BOOT,
        "arm_preprocessing": ARM_DESC,
        "phys_ckpt": PHYS_CKPT, "inst_ckpt": INST_CKPT, "decoder_ckpt": DECODER_CKPT,
        "tglc_path": TGLC_PATH, "phyts_path": PHYTS_PATH, "grid_range": GRID_RANGE,
        "git_commit": _git_commit(),
        "tic_sector_sha256": tic_sector_sha,
        "row_gaia_sector_sha256": _sha(gaia, sectors),
        "label_sha256": _sha(y),
        "split_sha256": _sha(np.concatenate([train_idx, test_idx])),
        "model_hashes_unchanged": bool(model_hashes_unchanged),
        "phyts_flux_used": False,
        "physics_config": {"readout": os.environ.get("JEPA_READOUT", "mean"),
                           "n_tokens": os.environ.get("JEPA_NTOKENS", "16")},
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"A(raw)={acc['A']:.4f}  B(native-clean)={acc['B']:.4f}  "
          f"C(grid-raw)={acc['C']:.4f}  D(grid-clean)={acc['D']:.4f}", flush=True)
    for k in ("B_minus_A", "C_minus_A", "D_minus_C", "D_minus_A"):
        c = boot[k]
        print(f"  {k}: {report['differences'][k]:+.4f}  95% CI [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}]",
              flush=True)
    print(f"wrote {OUT_DIR}/final_summary.json + per_curve_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
