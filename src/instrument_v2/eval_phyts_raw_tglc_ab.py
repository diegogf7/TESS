from __future__ import annotations
"""Matched A/B on RAW TGLC flux (not PhyTS flux): does subtracting the decoded
instrument help physics classification when the input is the SAME data domain the
instrument model was trained on?

PhyTS supplies ONLY (TIC, GaiaID, sector, label). The light curve is the raw TGLC
`aperture_flux` matched by (TIC, sector), quality-filtered with the central qclean
rule, then fed through both arms:

  ARM A  filtered raw TGLC -> physics preprocessing -> frozen physics JEPA
  ARM B  same filtered raw -> frozen instrument JEPA + direct decoder -> subtract
         (no scale fit) -> physics preprocessing -> same frozen physics JEPA

Same rows/order/TICs/labels/split both arms; frozen everything; KNN(20) probe.

  python -m src.instrument_v2.eval_phyts_raw_tglc_ab
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder, area_one_hot
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import BAD_TESS_MASK          # central qclean mask (16437)
from src.instrument_v2.eval_phyts_instrument_ab import (              # exact reuse
    DATA_PATH as PHYTS_PATH, PHYS_CKPT, INST_CKPT, DECODER_CKPT, GRID_RANGE,
    DEVICE, GRID, physics_grid, instrument_cleaned_curve, encode_physics,
    matched_split, assert_classes_present, classify, _freeze,
)
from src.instrument_v2.inspect_phyts_aperiodic_eclipse import predict_arm

TGLC_PATH = os.environ.get(
    "TGLC_PATH",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "phyts_raw_tglc_ab"))

# direct (default): 1024-output decoder. cbv: build_decoder(8) + fixed area bases,
# template = area_basis @ 8-weights. Everything after the template is identical.
CLEAN_MODE = os.environ.get("CLEAN_MODE", "direct")
assert CLEAN_MODE in ("direct", "cbv", "cbv_area_heads", "learned"), CLEAN_MODE
# AREA_CONDITIONED=1 (cbv only): use the area-conditioned weight decoder
# (decoder_input = concat(latent, area one-hot)) instead of the global one.
AREA_CONDITIONED = os.environ.get("AREA_CONDITIONED", "0") == "1"
AREA_DECODER = os.environ.get(
    "AREA_DECODER",
    "artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/"
    "single_star_weight_decode_area_conditioned/decoder.pth")


def quality_filter(time, flux, tess, tglc):
    """The SAME central qclean rule as sector14_dataset.grid_frame: keep finite
    cadences that are TESS-clean (BAD_TESS_MASK) and TGLC-clean. Removed cadences
    simply drop out (no interpolation)."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    tess = np.asarray(tess, dtype=np.int64)
    tglc = np.asarray(tglc, dtype=np.int64)
    good = (np.isfinite(time) & np.isfinite(flux)
            & ((tess & BAD_TESS_MASK) == 0) & (tglc == 0))
    return time[good], flux[good]


def match_phyts_tglc(phyts, tglc, phyts_gaia="phyts_gaia"):
    """Match each PhyTS row to its raw TGLC row by Gaia DR3 id + sector. TIC is
    NOT a unique key -- one TESS TIC can blend several Gaia sources, so
    (TIC, sector) is ambiguous; PhyTS's GaiaID pins the exact labelled star. The
    same Gaia source on two chips (duplicate GAIADR3) is de-duplicated (first)."""
    tglc = tglc.drop(columns=[c for c in ("TIC",) if c in tglc.columns])   # keep PhyTS TIC
    tglc = tglc.drop_duplicates(["GAIADR3", "sector"], keep="first")
    left = phyts.rename(columns={phyts_gaia: "GAIADR3"})
    merged = left.merge(tglc, on=["GAIADR3", "sector"], how="left", indicator=True)
    matched = merged[merged["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    unmatched = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return matched, unmatched


def ordered_hash_tic_sector(tics, sectors):
    h = hashlib.sha256()
    for t, s in zip(tics, sectors):
        h.update(f"{t}|{int(s)}\n".encode())
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    # --- PhyTS metadata ONLY (TIC, GaiaID, sector, label) --------------------
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next((c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id")
                     if c in phyts.columns), None)
    if gaia_col is None:
        raise RuntimeError("PhyTS has no Gaia id column -- cannot match TGLC by GaiaDR3")
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})
    n_phyts = len(phyts)
    expected = sorted(phyts["label"].unique())

    # --- raw TGLC (aperture_flux); match on (TIC, sector) --------------------
    tglc_cols = set(pq.read_schema(TGLC_PATH).names)
    flux_col = "aperture_flux" if "aperture_flux" in tglc_cols else "flux"  # dense_v2 stores raw as 'flux'
    tglc_want = ["TIC", "sector", "GAIADR3", "time", flux_col, "TESS_flags", "TGLC_flags"]
    tglc_want += [c for c in ("area", "camera", "ccd", "ra", "dec") if c in tglc_cols]  # cbv area
    tglc = pd.read_parquet(TGLC_PATH, columns=tglc_want)
    tglc["TIC"] = tglc["TIC"].astype(str)
    tglc = tglc.rename(columns={flux_col: "aperture_flux"})
    matched, unmatched = match_phyts_tglc(phyts, tglc)
    n_matched = len(matched)
    print(f"PhyTS s14 rows: {n_phyts} | matched to raw TGLC: {n_matched} "
          f"({100 * n_matched / max(1, n_phyts):.1f}%) | unmatched: {len(unmatched)}", flush=True)
    if len(unmatched):
        print("  unmatched TICs (first 20):",
              unmatched["TIC"].head(20).tolist(), flush=True)

    tics = matched["TIC"].to_numpy().astype(str)
    sectors = matched["sector"].to_numpy()
    gaia = matched["GAIADR3"].to_numpy()
    y = np.array([CLASS_TO_IDX[l] for l in matched["label"]], dtype=np.int64)
    present = np.unique(y)
    missing = set(expected) - set(CLASSES[c] for c in present)
    if missing:
        raise RuntimeError(f"class(es) absent after matching: {sorted(missing)}")

    # --- frozen models (identical to the A/B) --------------------------------
    phys = _freeze(build_latent_jepa().to(DEVICE))
    phys.load_state_dict(torch.load(PHYS_CKPT, map_location=DEVICE), strict=False)
    _freeze(phys)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                      readout="mean", predictor_type="mlp").to(DEVICE)
    inst.load_state_dict(torch.load(INST_CKPT, map_location=DEVICE))
    _freeze(inst)
    a2i = None
    if CLEAN_MODE == "cbv":                               # 8-weight decoder + fixed area bases
        from src.instrument_v2.run_tglc_physics_jepa_ab import (   # lazy: avoid import cycle
            WEIGHT_DECODER_CKPT, CBV_RANK, _resolve_bases_npz, _load_bases, _areas_for,
        )
        from src.instrument_v2.decode_single_star_k8 import area_index_map, load_area_decoder
        bases = _load_bases(_resolve_bases_npz())
        areas = _areas_for(matched)
        if AREA_CONDITIONED:                              # area decoder (one-hot OR area-heads; drop-in)
            decoder, a2i, kind = load_area_decoder(AREA_DECODER, DEVICE)
            _freeze(decoder)
            if a2i != area_index_map(sorted(bases)):
                raise RuntimeError("area map in checkpoint != deterministic sorted training-area map")
            print(f"CBV cleaning (AREA {kind}): {len(a2i)} areas, {AREA_DECODER}", flush=True)
        else:
            decoder = _freeze(build_decoder(CBV_RANK).to(DEVICE))
            decoder.load_state_dict(torch.load(WEIGHT_DECODER_CKPT, map_location=DEVICE))
            print(f"CBV cleaning: {CBV_RANK} weights x {len(bases)} area bases "
                  f"({_resolve_bases_npz()})", flush=True)
    else:
        decoder = _freeze(build_decoder(1024).to(DEVICE))
        decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
        bases = areas = None
    _freeze(decoder)
    hashes = {"phys": state_hash(phys), "inst_teacher": state_hash(inst.teacher),
              "inst_student": state_hash(inst.student),
              "inst_predictor": state_hash(inst.predictor), "decoder": state_hash(decoder)}

    # --- both arms from the SAME quality-filtered raw arrays ------------------
    A_X = np.zeros((n_matched, GRID), np.float32); A_M = np.zeros((n_matched, GRID), np.float32)
    B_X = np.zeros((n_matched, GRID), np.float32); B_M = np.zeros((n_matched, GRID), np.float32)
    for i in range(n_matched):
        ft, ff = quality_filter(matched["time"].iloc[i], matched["aperture_flux"].iloc[i],
                                matched["TESS_flags"].iloc[i], matched["TGLC_flags"].iloc[i])
        A_X[i], A_M[i] = physics_grid(ft, ff)
        basis = bases[int(areas[i])] if CLEAN_MODE == "cbv" else None    # this curve's area basis
        av = area_one_hot(a2i, int(areas[i])) if (CLEAN_MODE == "cbv" and AREA_CONDITIONED) else None
        ct, cf = instrument_cleaned_curve(ft, ff, inst, decoder, t0, t1, basis, av)   # no scale fit
        B_X[i], B_M[i] = physics_grid(ct, cf)
        if i % 200 == 0:
            print(f"  building arms {i}/{n_matched}", flush=True)
    lat_a = encode_physics(phys, A_X, A_M)
    lat_b = encode_physics(phys, B_X, B_M)
    assert np.isfinite(lat_a).all() and np.isfinite(lat_b).all(), "non-finite latents"

    assert state_hash(phys) == hashes["phys"], "physics changed"
    assert state_hash(inst.teacher) == hashes["inst_teacher"], "instrument changed"
    assert state_hash(inst.student) == hashes["inst_student"], "instrument student changed"
    assert state_hash(inst.predictor) == hashes["inst_predictor"], "instrument predictor changed"
    assert state_hash(decoder) == hashes["decoder"], "decoder changed"

    # --- one shared split, both arms -----------------------------------------
    train_idx, test_idx = matched_split(tics, y)
    assert_classes_present(y, train_idx, test_idx)
    base_acc, base_recall = classify(lat_a, y, train_idx, test_idx, present)
    clean_acc, clean_recall = classify(lat_b, y, train_idx, test_idx, present)
    base_pred = predict_arm(lat_a, y, train_idx, test_idx)
    clean_pred = predict_arm(lat_b, y, train_idx, test_idx)

    # --- per-curve predictions on held-out rows ------------------------------
    csv_rows = []
    for j, gi in enumerate(test_idx):
        csv_rows.append({
            "TIC": tics[gi], "GaiaID": gaia[gi], "sector": int(sectors[gi]),
            "true_label": CLASSES[y[gi]],
            "baseline_prediction": CLASSES[base_pred[j]],
            "cleaned_prediction": CLASSES[clean_pred[j]],
            "baseline_correct": int(base_pred[j] == y[gi]),
            "cleaned_correct": int(clean_pred[j] == y[gi])})
    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "per_curve_predictions.csv"), index=False)

    per_class_counts = {CLASSES[c]: int((y == c).sum()) for c in present}
    recall_diff = {c: clean_recall[c] - base_recall[c] for c in base_recall}
    report = {
        "n_phyts_sector14": int(n_phyts),
        "n_matched": int(n_matched),
        "pct_matched": round(100 * n_matched / max(1, n_phyts), 3),
        "n_unmatched": int(len(unmatched)),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "classes": [CLASSES[c] for c in present],
        "per_class_counts": per_class_counts,
        "baseline_balanced_accuracy": base_acc,
        "cleaned_balanced_accuracy": clean_acc,
        "cleaned_minus_baseline": clean_acc - base_acc,
        "baseline_per_class_recall": base_recall,
        "cleaned_per_class_recall": clean_recall,
        "per_class_recall_diff": recall_diff,
        "tic_sector_sha256": ordered_hash_tic_sector(tics, sectors),
        "flux_column_used": "aperture_flux (raw TGLC)",
        "quality_mask": BAD_TESS_MASK, "tglc_flags_removed": "nonzero",
        "matched_by": "GaiaDR3+sector (TIC is not unique -- blends)",
        "phys_ckpt": PHYS_CKPT, "inst_ckpt": INST_CKPT,
        "clean_mode": CLEAN_MODE,
        "area_conditioned": bool(CLEAN_MODE == "cbv" and AREA_CONDITIONED),
        "decoder_ckpt": (AREA_DECODER if (CLEAN_MODE == "cbv" and AREA_CONDITIONED)
                         else WEIGHT_DECODER_CKPT if CLEAN_MODE == "cbv" else DECODER_CKPT),
        "cbv_bases_npz": (_resolve_bases_npz() if CLEAN_MODE == "cbv" else None),
        "tglc_path": TGLC_PATH,
        "model_hashes_unchanged": True, "phyts_flux_used": False,
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"baseline={base_acc:.6f}  cleaned={clean_acc:.6f}  diff={clean_acc - base_acc:+.6f}",
          flush=True)
    print(f"wrote {OUT_DIR}/final_summary.json + per_curve_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
