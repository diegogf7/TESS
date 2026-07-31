from __future__ import annotations
"""Matched raw-vs-cleaned physics-JEPA PRETRAINING A/B on raw TGLC (Sector 14).

Train two LatentJEPAs from scratch, bit-identical init/split/batch-order/mask
seeds, differing ONLY in whether the decoded instrument template was subtracted
from the input curves. Then freeze both and probe the 2x2:

               raw eval curves     cleaned eval curves
  raw JEPA     PRIMARY raw         cross-domain control
  cleaned JEPA cross-domain        PRIMARY cleaned

Primary result = (cleaned JEPA on cleaned) - (raw JEPA on raw).

Stages:
  --stage prepare              build+cache matched raw/cleaned pretrain+eval arrays, per-seed init
  --stage train --arm raw      pretrain the raw JEPA   (array task 0)
  --stage train --arm cleaned  pretrain the cleaned JEPA(array task 1)
  --stage evaluate             frozen 2x2 KNN probe + collapse metrics

The instrument JEPA, its decoder, and the shared grid are frozen and never
updated. PhyTS supplies only labels/ids; PhyTS flux is never used. This shares NO
code with the fine-tuning experiment.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa, jepa_latent_loss
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder, load_area_decoder, area_one_hot
from src.shared_s4d.model import build_model as build_shared_model
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.sector14_dataset import grid_curve_shared, BAD_TESS_MASK
from src.instrument_v2.eval_phyts_instrument_ab import DEVICE, GRID, physics_grid, encode_physics
from src.instrument_v2.eval_phyts_raw_tglc_ab import (
    PHYTS_PATH, INST_CKPT, DECODER_CKPT, GRID_RANGE,
    match_phyts_tglc, quality_filter, ordered_hash_tic_sector,
)

# ---- paths / config ---------------------------------------------------------
PRETRAIN_PATH = os.environ.get(
    "PRETRAIN_PATH", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
EVAL_TGLC_PATH = os.environ.get(
    "EVAL_TGLC_PATH", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet")
PHYS_CKPT_DIR = os.environ.get(
    "PHYS_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/tglc_physics_jepa_ab")
# Where the RAW-arm checkpoint lives at evaluate time. Defaults to PHYS_CKPT_DIR;
# a cbv run points it at the existing experiment so the raw JEPA is REUSED, never
# retrained. The cleaned(=cbv) arm always comes from PHYS_CKPT_DIR.
RAW_CKPT_DIR = os.environ.get("RAW_CKPT_DIR", PHYS_CKPT_DIR)
OUT_DIR = os.environ.get("OUT_DIR", os.path.join("artifacts", "instrument_v2", "tglc_physics_jepa_ab"))
PREP_DIR = os.path.join(OUT_DIR, "prepared")

# ---- CBV-weight cleaning (CLEAN_MODE=cbv | cbv_area_heads) -------------------
# direct (default): existing 1024-output decoder, unchanged behaviour.
# cbv: global build_decoder(8) predicts 8 weights; template = area_basis @ weights.
# cbv_area_heads: 64-area-head decoder picks that area's 8-weight head instead.
# Everything after the template is identical (subtract_native + one resample).
CLEAN_MODE = os.environ.get("CLEAN_MODE", "direct")
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_qclean_v1"))
WEIGHT_DECODER_CKPT = os.environ.get(
    "WEIGHT_DECODER_CKPT", os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
# cbv_area_heads: route CBV cleaning through the 64-area-head decoder (latent +
# area one-hot -> that area's 8-weight head), NOT the global 8-weight decoder.
AREA_DECODER_CKPT = os.environ.get(
    "AREA_DECODER_CKPT", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_heads", "decoder.pth"))
# learned: shared-S4D autoencoder that predicts the 1024 systematics DIRECTLY from
# one curve (no instrument JEPA, no CBV bases). LEARNED_CKPT = its _best.pth.
LEARNED_CKPT = os.environ.get("LEARNED_CKPT", "")
assert CLEAN_MODE in ("direct", "cbv", "cbv_area_heads", "learned"), CLEAN_MODE
IS_CBV = CLEAN_MODE in ("cbv", "cbv_area_heads")            # both use area CBV bases
AREA_HEADS = CLEAN_MODE == "cbv_area_heads"
LEARNED = CLEAN_MODE == "learned"
CBV_BASES_NPZ = os.environ.get("CBV_BASES_NPZ", "")     # explicit path; else unique glob below
CBV_RANK = 8

EXPECTED_TIC_SECTOR_SHA = "6a1796a05d4313479a39cf8fdf5e8b273544558c28692fdbba9df63dc8f425cc"
EXPECTED_N_EVAL = 2409
EXPECTED_N_PRETRAIN = 123858
MIN_GOOD = 64                     # drop degenerate pretraining curves (unlabeled) below this many cadences
EPOCHS = 30
BATCH_SIZE = 256
LR = 3e-4
VAR_WEIGHT = 0.05
CONFIG_KEYS = ["pretrain_sig", "eval_tglc_sig", "inst_sig", "decoder_sig", "grid_sig", "bad_tess_mask",
               "clean_mode", "area_decoder_sig", "learned_sig"]


# ---- small hashing helpers --------------------------------------------------
def _file_sig(path):
    if not path or not os.path.exists(path):     # learned mode ignores the instrument/CBV paths
        return None
    return hashlib.sha256(f"{os.path.abspath(path)}|{os.stat(path).st_size}".encode()).hexdigest()[:16]


def _content_hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _ordered_hash(*cols):
    h = hashlib.sha256()
    for row in zip(*cols):
        h.update(("|".join(str(v) for v in row) + "\n").encode())
    return h.hexdigest()


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def current_config():
    return {"pretrain_path": PRETRAIN_PATH, "pretrain_sig": _file_sig(PRETRAIN_PATH),
            "eval_tglc_path": EVAL_TGLC_PATH, "eval_tglc_sig": _file_sig(EVAL_TGLC_PATH),
            "phyts_path": PHYTS_PATH, "inst_ckpt": INST_CKPT, "inst_sig": _file_sig(INST_CKPT),
            "decoder_ckpt": DECODER_CKPT, "decoder_sig": _file_sig(DECODER_CKPT),
            "grid_range": GRID_RANGE, "grid_sig": _content_hash(GRID_RANGE),
            "bad_tess_mask": int(BAD_TESS_MASK), "flux_policy": "raw aperture_flux/flux only",
            "clean_mode": CLEAN_MODE,
            "area_decoder_sig": (_file_sig(AREA_DECODER_CKPT) if AREA_HEADS else None),
            "learned_sig": (_file_sig(LEARNED_CKPT) if LEARNED else None)}


def assert_manifest_matches(manifest, current):
    """Refuse stale prepared data: every config key must match the current run."""
    for k in CONFIG_KEYS:
        if manifest.get(k) != current.get(k):
            raise RuntimeError(f"stale prepared data: '{k}' changed "
                               f"({manifest.get(k)} != {current.get(k)}) -- re-run --stage prepare")


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def strict_load(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"strict load failed: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    return model


# ---- arm construction (native cleaning, batched instrument decode) ----------
def _grid_times(t0, t1):
    return t0 + (np.arange(GRID) + 0.5) / GRID * (t1 - t0)


def grid_for_instrument(ft, ff, t0, t1):
    """Per-curve: median/MAD normalize the native flux, map to the shared S14
    instrument grid. Returns X, M, median, scale, valid mask."""
    normed, med, mad = normalize_median_mad(ff)
    scale = 1.4826 * mad if 1.4826 * mad > 0 else 1.0
    X, M = grid_curve_shared(ft, normed, t0, t1, GRID)
    return X.astype(np.float32), M.astype(np.float32), float(med), float(scale), M > 0


def subtract_native(ft, ff, decoded, valid, scale, grid_times):
    """Subtract the decoded instrument template at the NATIVE cadences, in flux
    units, no fitted coefficient. decoded=None (too few grid bins) -> no cleaning.
    Numerically identical to the fine-tuning cleaned_native_flux back-half."""
    if decoded is None:
        return np.asarray(ff, np.float64)
    dec_native = np.interp(np.asarray(ft, float), grid_times[valid], decoded[valid])
    return np.asarray(ff, np.float64) - dec_native * scale


def batched_decode(inst, decoder, Xg, Mg, batch=512):
    """One batched GPU pass (NOT one-per-star) to decode the instrument template
    for every gridded curve."""
    decoded = np.zeros_like(Xg)
    with torch.no_grad():
        for s in range(0, len(Xg), batch):
            f = torch.tensor(Xg[s:s + batch], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mg[s:s + batch], dtype=torch.float32, device=DEVICE)
            z = inst.encode(f, m, view="predicted")
            decoded[s:s + batch] = decoder(z).detach().cpu().numpy()
    return decoded


def batched_cbv_templates(inst, wdecoder, Xg, Mg, areas, bases, batch=512, a2i=None):
    """Frozen instrument JEPA -> 8 CBV weights -> template = area_basis @ weights,
    per curve. Returns the SAME (n, 1024) template array batched_decode would, so
    the rest of build_arms is unchanged.

    a2i is None (cbv)          : GLOBAL 8-weight decoder on the bare latent.
    a2i given (cbv_area_heads) : AREA-HEAD decoder -- feed concat(latent, area
                                 one-hot); its argmax selects that area's head.
    A missing area basis, or an area unknown to the area-head decoder, HARD-FAILS
    -- never a silent raw or global fallback."""
    B = np.zeros((len(Xg), GRID, CBV_RANK), np.float32)
    onehots = np.zeros((len(Xg), len(a2i)), np.float32) if a2i is not None else None
    for i, a in enumerate(areas):
        if int(a) not in bases:
            raise RuntimeError(f"no CBV basis for area {int(a)} -- refusing to fall back to raw")
        B[i] = bases[int(a)]
        if a2i is not None:
            onehots[i] = area_one_hot(a2i, int(a))                     # unknown area HARD-FAILS
    templates = np.zeros_like(Xg)
    with torch.no_grad():
        for s in range(0, len(Xg), batch):
            f = torch.tensor(Xg[s:s + batch], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mg[s:s + batch], dtype=torch.float32, device=DEVICE)
            z = inst.encode(f, m, view="predicted")
            if a2i is None:
                w = wdecoder(z)                                        # global: (b, 8)
            else:
                oh = torch.tensor(onehots[s:s + batch], dtype=torch.float32, device=DEVICE)
                w = wdecoder(torch.cat([z.reshape(z.shape[0], -1), oh], dim=1))   # area head: (b, 8)
            assert w.shape[1] == CBV_RANK, f"expected {CBV_RANK} weights/curve, got {w.shape[1]}"
            bb = torch.tensor(B[s:s + batch], dtype=torch.float32, device=DEVICE)
            tmpl = torch.einsum("bk,blk->bl", w, bb).detach().cpu().numpy()
            assert tmpl.shape[1] == GRID, f"template must be {GRID} points, got {tmpl.shape[1]}"
            templates[s:s + batch] = tmpl
    return templates


def batched_learned_templates(model, Xg, Mg, batch=512):
    """CLEAN_MODE=learned: shared-S4D autoencoder predicts the 1024-cadence
    systematics DIRECTLY from each single curve (no instrument JEPA, no CBV
    bases). Returns the SAME (n, 1024) template array batched_decode would."""
    templates = np.zeros_like(Xg)
    with torch.no_grad():
        for s in range(0, len(Xg), batch):
            x = torch.tensor(Xg[s:s + batch], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mg[s:s + batch], dtype=torch.float32, device=DEVICE)
            s_hat, _ = model(x, m)
            assert s_hat.shape[1] == GRID, f"expected {GRID}-pt systematics, got {s_hat.shape[1]}"
            templates[s:s + batch] = s_hat.detach().cpu().numpy()
    return templates


def build_arms(times, fluxes, inst, decoder, t0, t1, areas=None, bases=None, a2i=None,
               learned_model=None):
    """From filtered native (time, flux) lists build matched raw and cleaned
    physics inputs. Decode once (batched), then subtract on native timestamps and
    run the physics preprocessing exactly once per arm.

    bases is None (direct): `decoder` is the 1024-output instrument decoder.
    bases given (cbv): `decoder` is build_decoder(8); the template is
    area_basis @ 8-weights. Only the template SOURCE differs -- subtraction,
    scaling, native interpolation and the single physics resample are identical."""
    n = len(times)
    grid_times = _grid_times(t0, t1)
    Xg = np.zeros((n, GRID), np.float32); Mg = np.zeros((n, GRID), np.float32)
    med = np.zeros(n); scale = np.zeros(n); valids = []
    for i in range(n):
        Xg[i], Mg[i], med[i], scale[i], v = grid_for_instrument(times[i], fluxes[i], t0, t1)
        valids.append(v)
    if learned_model is not None:
        decoded = batched_learned_templates(learned_model, Xg, Mg)
    elif bases is not None:
        decoded = batched_cbv_templates(inst, decoder, Xg, Mg, areas, bases, a2i=a2i)
    else:
        decoded = batched_decode(inst, decoder, Xg, Mg)

    raw_X = np.zeros((n, GRID), np.float32); raw_M = np.zeros((n, GRID), np.float32)
    cln_X = np.zeros((n, GRID), np.float32); cln_M = np.zeros((n, GRID), np.float32)
    for i in range(n):
        ft, ff = times[i], fluxes[i]
        raw_X[i], raw_M[i] = physics_grid(ft, ff)
        dec = decoded[i] if valids[i].sum() >= 8 else None
        cln_X[i], cln_M[i] = physics_grid(ft, subtract_native(ft, ff, dec, valids[i], scale[i], grid_times))
        if i % 5000 == 0:
            print(f"    build_arms {i}/{n}", flush=True)
    assert np.array_equal(raw_M, cln_M), "raw and cleaned masks differ"
    return raw_X, raw_M, cln_X, cln_M


def assert_no_eval_overlap(df, excl):
    """Hard-fail if ANY (GaiaDR3, sector) in df is a labeled eval-cohort star."""
    overlap = set(zip(df["GAIADR3"].tolist(), df["sector"].tolist())) & set(excl)
    if overlap:
        raise RuntimeError(f"{len(overlap)} eval GaiaDR3+sector present in pretraining -- exclusion failed")


def remove_eval_cohort(df, excl):
    """Drop pretraining rows whose (GaiaDR3, sector) is a labeled eval-cohort
    star, then assert none remain (evaluation stars must never pretrain)."""
    key = list(zip(df["GAIADR3"].tolist(), df["sector"].tolist()))
    kept = df[np.array([k not in excl for k in key], dtype=bool)].reset_index(drop=True)
    assert_no_eval_overlap(kept, excl)
    return kept


def _load_curves(path):
    cols = set(pq.read_schema(path).names)
    flux_col = "aperture_flux" if "aperture_flux" in cols else "flux"
    want = ["TIC", "sector", "GAIADR3", "time", flux_col, "TESS_flags", "TGLC_flags"]
    want += [c for c in ("area", "camera", "ccd", "ra", "dec") if c in cols]  # cbv: area lookup
    df = pd.read_parquet(path, columns=want)
    df["TIC"] = df["TIC"].astype(str)
    return df.rename(columns={flux_col: "flux_raw"})


def _resolve_bases_npz():
    """The one cached train-TIC area-basis npz (the existing K=8 bases; never
    rebuilt here). Absence or ambiguity hard-fails."""
    if CBV_BASES_NPZ:
        if not os.path.exists(CBV_BASES_NPZ):
            raise RuntimeError(f"CBV_BASES_NPZ does not exist: {CBV_BASES_NPZ}")
        return CBV_BASES_NPZ
    hits = sorted(glob.glob(os.path.join(
        GROUP_ART_DIR, f"area_group_cbv_r{CBV_RANK}_g32_mv16_q16437_tglc0_*.npz")))
    if len(hits) != 1:
        raise RuntimeError(f"need exactly one basis npz in {GROUP_ART_DIR}, found "
                           f"{len(hits)} -- set CBV_BASES_NPZ explicitly")
    return hits[0]


def _load_bases(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    bases = {int(a): d[f"B_{int(a)}"].astype(np.float32) for a in d["areas"]}
    for a, B in bases.items():
        if B.shape != (GRID, CBV_RANK):
            raise RuntimeError(f"area {a} basis shape {B.shape} != ({GRID}, {CBV_RANK})")
    return bases


def _areas_for(df):
    """Per-row area for the CBV basis. Uses the existing `area` column, else the
    existing add_area(camera,ccd,ra,dec) assignment. Invalid areas hard-fail."""
    if "area" in df.columns and df["area"].notna().all():
        a = df["area"].astype(int).to_numpy()
    elif {"camera", "ccd", "ra", "dec"} <= set(df.columns):
        from src.regions.areas import add_area
        a = add_area(df.copy())["area"].astype(int).to_numpy()
    else:
        raise RuntimeError("no `area` column nor camera/ccd/ra/dec -- cannot assign CBV basis")
    if (a == -1).any():
        raise RuntimeError(f"{int((a == -1).sum())} rows have area -1 -- cannot clean without a basis")
    return a


def _filter_curves(df, min_good):
    """Quality-filter each curve; return kept native (time,flux) + row metadata."""
    times, fluxes, keep = [], [], []
    for i in range(len(df)):
        ft, ff = quality_filter(df["time"].iloc[i], df["flux_raw"].iloc[i],
                                df["TESS_flags"].iloc[i], df["TGLC_flags"].iloc[i])
        if len(ft) >= min_good:
            times.append(ft); fluxes.append(ff); keep.append(i)
    sub = df.iloc[keep].reset_index(drop=True)
    return times, fluxes, sub


# ---- stage: prepare ---------------------------------------------------------
def stage_prepare(seed):
    os.makedirs(PREP_DIR, exist_ok=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    manifest_path = os.path.join(PREP_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        saved = json.load(open(manifest_path))
        try:
            assert_manifest_matches(saved, current_config())
            if CLEAN_MODE == "cbv" and (               # cbv artifacts not in CONFIG_KEYS
                    saved.get("clean_mode") != "cbv"
                    or saved.get("weight_decoder_sig") != _file_sig(WEIGHT_DECODER_CKPT)
                    or saved.get("bases_sig") != _file_sig(_resolve_bases_npz())):
                raise RuntimeError("cbv weight-decoder or bases changed")
            if AREA_HEADS and (                        # area-head artifacts + bases
                    saved.get("clean_mode") != "cbv_area_heads"
                    or saved.get("area_decoder_sig") != _file_sig(AREA_DECODER_CKPT)
                    or saved.get("bases_sig") != _file_sig(_resolve_bases_npz())):
                raise RuntimeError("area-head decoder or bases changed")
            if LEARNED and (                           # shared-S4D cleaner
                    saved.get("clean_mode") != "learned"
                    or saved.get("learned_sig") != _file_sig(LEARNED_CKPT)):
                raise RuntimeError("learned model changed")
            print("prepared data already valid; ensuring per-seed init only", flush=True)
            _ensure_init(seed)
            return
        except RuntimeError as e:
            print(f"manifest mismatch ({e}); rebuilding", flush=True)

    # --- eval cohort: PhyTS labels matched to raw TGLC by GaiaDR3+sector ------
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next((c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id")
                     if c in phyts.columns), None)
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})
    eval_raw = _load_curves(EVAL_TGLC_PATH).rename(columns={"flux_raw": "aperture_flux"})
    matched, _ = match_phyts_tglc(phyts, eval_raw)
    matched = matched.rename(columns={"aperture_flux": "flux_raw"})
    if len(matched) != EXPECTED_N_EVAL:
        raise RuntimeError(f"eval cohort n={len(matched)} != {EXPECTED_N_EVAL}")
    eval_tics = matched["TIC"].to_numpy().astype(str)
    eval_sectors = matched["sector"].to_numpy()
    eval_gaia = matched["GAIADR3"].to_numpy()
    if ordered_hash_tic_sector(eval_tics, eval_sectors) != EXPECTED_TIC_SECTOR_SHA:
        raise RuntimeError("eval cohort TIC/sector hash != prior frozen experiment")
    eval_y = np.array([CLASS_TO_IDX[l] for l in matched["label"]], dtype=np.int64)
    if len(np.unique(eval_y)) != 7:
        raise RuntimeError("eval cohort missing a class")
    excl = set(zip(eval_gaia.tolist(), eval_sectors.tolist()))     # (GaiaDR3, sector) to exclude
    exclusion_hash = _ordered_hash(sorted(f"{g}|{s}" for g, s in excl))

    # --- pretraining set: dense_v2 MINUS the eval cohort ---------------------
    pre = _load_curves(PRETRAIN_PATH)
    before = len(pre)
    pre = remove_eval_cohort(pre, excl)                    # hard-fails if any eval star remains
    print(f"pretrain rows: {before} -> {len(pre)} after excluding {before - len(pre)} eval cohort", flush=True)

    pre_times, pre_fluxes, pre = _filter_curves(pre, MIN_GOOD)
    print(f"pretrain curves after quality/min-good filter: {len(pre)}", flush=True)

    # --- frozen cleaner ------------------------------------------------------
    # learned: shared-S4D autoencoder ONLY (no instrument JEPA / no CBV decoder).
    inst = decoder = learned_model = bases = pre_areas = a2i = None
    if LEARNED:
        if not LEARNED_CKPT or not os.path.exists(LEARNED_CKPT):
            raise RuntimeError(f"CLEAN_MODE=learned needs LEARNED_CKPT; got {LEARNED_CKPT!r}")
        lck = torch.load(LEARNED_CKPT, map_location=DEVICE)
        learned_model = build_shared_model().to(DEVICE)
        learned_model.load_state_dict(lck["model"] if isinstance(lck, dict) and "model" in lck else lck)
        learned_model.eval()
        for p in learned_model.parameters():
            p.requires_grad_(False)
        frozen_before = state_hash(learned_model)
        print(f"LEARNED cleaning: shared-S4D autoencoder {LEARNED_CKPT} "
              f"(direct 1024 systematics per curve)", flush=True)
    else:
        inst = strict_load(FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                                      readout="mean", predictor_type="mlp").to(DEVICE),
                           torch.load(INST_CKPT, map_location=DEVICE)).eval()
        for p in inst.parameters():
            p.requires_grad_(False)
        # direct: 1024-output decoder. cbv: global build_decoder(8). cbv_area_heads:
        # 64-area-head decoder (latent + area one-hot -> that area's 8-weight head).
        if CLEAN_MODE == "cbv_area_heads":
            decoder, a2i, kind = load_area_decoder(AREA_DECODER_CKPT, DEVICE)
            if kind != "area_heads":
                raise RuntimeError(f"expected an area_heads decoder at {AREA_DECODER_CKPT}, got kind={kind}")
            bases = _load_bases(_resolve_bases_npz())
            pre_areas = _areas_for(pre)
            print(f"CBV AREA-HEAD cleaning: {len(a2i)} heads x {len(bases)} area bases, "
                  f"decoder {AREA_DECODER_CKPT} ({_resolve_bases_npz()})", flush=True)
        elif CLEAN_MODE == "cbv":
            decoder = strict_load(build_decoder(CBV_RANK).to(DEVICE),
                                  torch.load(WEIGHT_DECODER_CKPT, map_location=DEVICE)).eval()
            bases = _load_bases(_resolve_bases_npz())
            pre_areas = _areas_for(pre)
            print(f"CBV cleaning: {CBV_RANK} weights x {len(bases)} area bases "
                  f"({_resolve_bases_npz()})", flush=True)
        else:
            decoder = strict_load(build_decoder(1024).to(DEVICE),
                                  torch.load(DECODER_CKPT, map_location=DEVICE)).eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        frozen_before = (state_hash(inst.teacher), state_hash(inst.student),
                         state_hash(inst.predictor), state_hash(decoder))

    # --- build matched arms (pretrain + eval) --------------------------------
    print("building pretraining arms...", flush=True)
    pre_raw_X, pre_raw_M, pre_cln_X, pre_cln_M = build_arms(
        pre_times, pre_fluxes, inst, decoder, t0, t1, pre_areas, bases, a2i, learned_model)
    eval_times, eval_fluxes, matched2 = _filter_curves(matched, min_good=1)
    if len(matched2) != EXPECTED_N_EVAL:
        raise RuntimeError("an eval curve was dropped by the min-good filter")
    eval_areas = _areas_for(matched2) if IS_CBV else None
    print("building evaluation arms...", flush=True)
    ev_raw_X, ev_raw_M, ev_cln_X, ev_cln_M = build_arms(
        eval_times, eval_fluxes, inst, decoder, t0, t1, eval_areas, bases, a2i, learned_model)

    frozen_after = (state_hash(learned_model) if LEARNED else
                    (state_hash(inst.teacher), state_hash(inst.student),
                     state_hash(inst.predictor), state_hash(decoder)))
    if frozen_before != frozen_after:
        raise RuntimeError("cleaner model changed during preparation")

    # --- one deterministic TIC-disjoint pretraining/validation split ---------
    pre_tics = pre["TIC"].to_numpy().astype(str)
    dummy = np.zeros(len(pre_tics))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    tr, va = next(gss.split(dummy, dummy, groups=pre_tics))

    np.save(os.path.join(PREP_DIR, "pretrain_raw_X.npy"), pre_raw_X)
    np.save(os.path.join(PREP_DIR, "pretrain_raw_M.npy"), pre_raw_M)
    np.save(os.path.join(PREP_DIR, "pretrain_cleaned_X.npy"), pre_cln_X)
    np.save(os.path.join(PREP_DIR, "pretrain_cleaned_M.npy"), pre_cln_M)
    np.savez(os.path.join(PREP_DIR, "pretrain_meta.npz"), tics=pre_tics,
             gaia=pre["GAIADR3"].to_numpy(), sectors=pre["sector"].to_numpy(),
             train_idx=tr, val_idx=va)
    np.save(os.path.join(PREP_DIR, "eval_raw_X.npy"), ev_raw_X)
    np.save(os.path.join(PREP_DIR, "eval_raw_M.npy"), ev_raw_M)
    np.save(os.path.join(PREP_DIR, "eval_cleaned_X.npy"), ev_cln_X)
    np.save(os.path.join(PREP_DIR, "eval_cleaned_M.npy"), ev_cln_M)
    np.savez(os.path.join(PREP_DIR, "eval_meta.npz"), tics=eval_tics, gaia=eval_gaia,
             sectors=eval_sectors, y=eval_y)

    manifest = {**current_config(),
                "git_commit": _git_commit(),
                "n_pretrain": int(len(pre)), "n_eval": int(EXPECTED_N_EVAL),
                "n_pretrain_train": int(len(tr)), "n_pretrain_val": int(len(va)),
                "pretrain_gaia_tic_sha256": _ordered_hash(pre["GAIADR3"].to_numpy(), pre_tics),
                "eval_gaia_tic_sha256": _ordered_hash(eval_gaia, eval_tics),
                "eval_tic_sector_sha256": ordered_hash_tic_sector(eval_tics, eval_sectors),
                "exclusion_hash": exclusion_hash,
                "instrument_decoder_unchanged": True,
                "clean_mode": CLEAN_MODE,
                "decoder_kind": ("learned_s4d" if LEARNED else "area_heads" if AREA_HEADS else
                                 ("cbv_global" if CLEAN_MODE == "cbv" else "direct")),
                "weight_decoder_sig": (_file_sig(WEIGHT_DECODER_CKPT) if CLEAN_MODE == "cbv" else None),
                "weight_decoder_ckpt": (WEIGHT_DECODER_CKPT if CLEAN_MODE == "cbv" else None),
                "area_decoder_ckpt": (AREA_DECODER_CKPT if AREA_HEADS else None),
                "area_decoder_sig": (_file_sig(AREA_DECODER_CKPT) if AREA_HEADS else None),
                "n_heads": (len(a2i) if AREA_HEADS else None),
                "learned_ckpt": (LEARNED_CKPT if LEARNED else None),
                "learned_sig": (_file_sig(LEARNED_CKPT) if LEARNED else None),
                "bases_sig": (_file_sig(_resolve_bases_npz()) if IS_CBV else None),
                "bases_npz": (_resolve_bases_npz() if IS_CBV else None),
                "quality_filter": {"bad_tess_mask": int(BAD_TESS_MASK), "tglc_flags": "== 0",
                                   "finite": True, "min_good_pretrain": MIN_GOOD}}
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote prepared arrays + manifest to {PREP_DIR}", flush=True)
    _ensure_init(seed)


def _ensure_init(seed):
    """Create one deterministic initial physics-JEPA state per seed (both arms
    later load it strictly, so they start bit-identically)."""
    init_path = os.path.join(PHYS_CKPT_DIR, f"physics_jepa_init_s{seed}.pth")
    if os.path.exists(init_path):
        return init_path
    torch.manual_seed(seed)
    model = build_latent_jepa()
    _atomic_save(model.state_dict(), init_path)
    print(f"wrote initial JEPA state -> {init_path} (hash {state_hash(model)[:12]})", flush=True)
    return init_path


def _require_valid_prepared():
    manifest_path = os.path.join(PREP_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError("no prepared data; run --stage prepare first")
    manifest = json.load(open(manifest_path))
    assert_manifest_matches(manifest, current_config())
    return manifest


# ---- stage: train -----------------------------------------------------------
def stage_train(arm, seed):
    assert arm in ("raw", "cleaned", "cbv")   # cbv reuses this path verbatim, loads pretrain_cbv_*
    manifest = _require_valid_prepared()
    meta = np.load(os.path.join(PREP_DIR, "pretrain_meta.npz"))
    train_idx, val_idx = meta["train_idx"], meta["val_idx"]
    X = torch.from_numpy(np.load(os.path.join(PREP_DIR, f"pretrain_{arm}_X.npy")))
    M = torch.from_numpy(np.load(os.path.join(PREP_DIR, f"pretrain_{arm}_M.npy")))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_latent_jepa().to(DEVICE)
    init_path = os.path.join(PHYS_CKPT_DIR, f"physics_jepa_init_s{seed}.pth")
    if not os.path.exists(init_path):
        raise RuntimeError(f"missing {init_path}; run --stage prepare SEED={seed}")
    model.load_state_dict(torch.load(init_path, map_location=DEVICE), strict=True)  # identical start
    init_hash = state_hash(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    perm_rng = np.random.default_rng(seed)                 # batch order: identical across arms
    ckpt_path = os.path.join(PHYS_CKPT_DIR, f"physics_jepa_{arm}_s{seed}_best.pth")
    best_val, best_epoch, best_std = float("inf"), -1, 0.0

    for epoch in range(EPOCHS):
        model.train()
        order = train_idx[perm_rng.permutation(len(train_idx))]
        tot = 0.0
        for s in range(0, len(order), BATCH_SIZE):
            j = torch.from_numpy(order[s:s + BATCH_SIZE])
            flux, mask = X[j].to(DEVICE), M[j].to(DEVICE)
            optimizer.zero_grad()
            pred, tgt, seg = model(flux, mask)             # segment masks: global RNG (same both arms)
            loss = jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT)
            loss.backward()
            optimizer.step()
            model.update_target()
            tot += loss.item()
        scheduler.step()

        torch.manual_seed(10_000 + seed)                   # deterministic, arm-identical validation masks
        model.eval()
        vtot, chunks = 0.0, []
        with torch.no_grad():
            for s in range(0, len(val_idx), BATCH_SIZE):
                j = torch.from_numpy(val_idx[s:s + BATCH_SIZE])
                flux, mask = X[j].to(DEVICE), M[j].to(DEVICE)
                pred, tgt, seg = model(flux, mask)
                vtot += jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT).item() * len(j)
                z = model.encode(flux, mask)
                chunks.append(z.reshape(len(j), -1).cpu())
        avg_val = vtot / len(val_idx)
        latent_std = float(torch.cat(chunks).std(0).mean())
        print(f"[{arm} s{seed}] epoch {epoch+1}/{EPOCHS} train {tot/max(1,len(order)//BATCH_SIZE):.5f} "
              f"val {avg_val:.5f} latent_std {latent_std:.4f}", flush=True)
        if avg_val < best_val:
            best_val, best_epoch, best_std = avg_val, epoch + 1, latent_std
            _atomic_save(model.state_dict(), ckpt_path)

    meta_out = {"arm": arm, "seed": seed, "val_loss": best_val, "best_epoch": best_epoch,
                "latent_std": best_std, "init_hash": init_hash, "ckpt": ckpt_path,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "var_weight": VAR_WEIGHT,
                "config": {"n_tokens": os.environ.get("JEPA_NTOKENS", "16"),
                           "readout": os.environ.get("JEPA_READOUT", "mean"),
                           "predictor": os.environ.get("JEPA_PREDICTOR", "transformer"),
                           "mask_ratio": os.environ.get("JEPA_MASK_RATIO", "0.5")}}
    with open(os.path.join(PHYS_CKPT_DIR, f"physics_jepa_{arm}_s{seed}_meta.json"), "w") as fh:
        json.dump(meta_out, fh, indent=2)
    print(f"[{arm} s{seed}] best val {best_val:.5f} @epoch {best_epoch} -> {ckpt_path}", flush=True)


# ---- stage: evaluate --------------------------------------------------------
def _classify(latents, y, train_idx, test_idx, present):
    scaler = StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=20).fit(scaler.fit_transform(latents[train_idx]), y[train_idx])
    pred = knn.predict(scaler.transform(latents[test_idx]))
    acc = float(balanced_accuracy_score(y[test_idx], pred))
    rec = recall_score(y[test_idx], pred, labels=present, average=None, zero_division=0)
    return acc, {CLASSES[c]: float(r) for c, r in zip(present, rec)}, pred


def _effective_rank(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stage_evaluate(seed):
    manifest = _require_valid_prepared()
    em = np.load(os.path.join(PREP_DIR, "eval_meta.npz"), allow_pickle=True)
    y, tics, gaia, sectors = em["y"], em["tics"].astype(str), em["gaia"], em["sectors"]
    present = np.unique(y)
    X = {k: np.load(os.path.join(PREP_DIR, f"eval_{k}_X.npy")) for k in ("raw", "cleaned")}
    Mk = {k: np.load(os.path.join(PREP_DIR, f"eval_{k}_M.npy")) for k in ("raw", "cleaned")}

    jepas, metas = {}, {}
    for arm in ("raw", "cleaned"):
        ckpt_dir = RAW_CKPT_DIR if arm == "raw" else PHYS_CKPT_DIR   # raw reused; cleaned=cbv here
        ck = os.path.join(ckpt_dir, f"physics_jepa_{arm}_s{seed}_best.pth")
        if not os.path.exists(ck):
            raise RuntimeError(f"missing checkpoint {ck}; train the {arm} arm first")
        m = build_latent_jepa().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        jepas[arm] = m
        mp = os.path.join(ckpt_dir, f"physics_jepa_{arm}_s{seed}_meta.json")
        metas[arm] = json.load(open(mp)) if os.path.exists(mp) else {}
    started_identical = (metas.get("raw", {}).get("init_hash") == metas.get("cleaned", {}).get("init_hash")
                         and metas.get("raw", {}).get("init_hash") is not None)

    # one shared TIC-disjoint split; all four cells use identical rows/indices
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(len(tics)), y, groups=tics))

    cells = {"raw_jepa_on_raw": ("raw", "raw"), "raw_jepa_on_cleaned": ("raw", "cleaned"),
             "cleaned_jepa_on_raw": ("cleaned", "raw"), "cleaned_jepa_on_cleaned": ("cleaned", "cleaned")}
    acc, recall, preds, collapse = {}, {}, {}, {}
    for name, (jm, dm) in cells.items():
        lat = encode_physics(jepas[jm], X[dm], Mk[dm])
        acc[name], recall[name], preds[name] = _classify(lat, y, tr, te, present)
        collapse[name] = {"latent_std": float(lat.std(0).mean()), "effective_rank": _effective_rank(lat)}

    primary = acc["cleaned_jepa_on_cleaned"] - acc["raw_jepa_on_raw"]

    rows = []
    for j, gi in enumerate(te):
        r = {"TIC": tics[gi], "GaiaDR3": gaia[gi], "true_label": CLASSES[y[gi]], "split": "test"}
        for name in cells:
            r[f"pred_{name}"] = CLASSES[preds[name][j]]
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_curve_predictions.csv"), index=False)

    summary = {
        "balanced_accuracy": acc,
        "primary_cleaned_minus_raw": primary,
        "per_class_recall": recall,
        "collapse_metrics": collapse,
        "n_pretrain": manifest["n_pretrain"], "n_pretrain_train": manifest["n_pretrain_train"],
        "n_pretrain_val": manifest["n_pretrain_val"],
        "n_eval": int(len(tics)), "n_eval_train": int(len(tr)), "n_eval_test": int(len(te)),
        "classes": [CLASSES[c] for c in present],
        "seed": seed,
        "clean_mode": manifest.get("clean_mode", CLEAN_MODE),
        "raw_ckpt": os.path.join(RAW_CKPT_DIR, f"physics_jepa_raw_s{seed}_best.pth"),
        "cleaned_ckpt": os.path.join(PHYS_CKPT_DIR, f"physics_jepa_cleaned_s{seed}_best.pth"),
        "pretrain_val_loss": {a: metas[a].get("val_loss") for a in ("raw", "cleaned")},
        "paths": {"pretrain": PRETRAIN_PATH, "eval_tglc": EVAL_TGLC_PATH, "phyts": PHYTS_PATH,
                  "inst_ckpt": INST_CKPT, "decoder_ckpt": DECODER_CKPT, "grid_range": GRID_RANGE,
                  "weight_decoder_ckpt": manifest.get("weight_decoder_ckpt"),
                  "bases_npz": manifest.get("bases_npz")},
        "eval_tic_sector_sha256": ordered_hash_tic_sector(tics, sectors),
        "eval_split_sha256": _ordered_hash(np.concatenate([tr, te])),
        "exclusion_hash": manifest["exclusion_hash"],
        "eval_excluded_from_pretraining": True,
        "instrument_decoder_unchanged": bool(manifest.get("instrument_decoder_unchanged", False)),
        "jepas_started_identical": bool(started_identical),
        "phyts_flux_used": False,
        "git_commit": _git_commit(),
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("2x2 balanced accuracy:", flush=True)
    print(f"  raw JEPA     : raw {acc['raw_jepa_on_raw']:.4f} | cleaned {acc['raw_jepa_on_cleaned']:.4f}", flush=True)
    print(f"  cleaned JEPA : raw {acc['cleaned_jepa_on_raw']:.4f} | cleaned {acc['cleaned_jepa_on_cleaned']:.4f}", flush=True)
    print(f"PRIMARY cleaned-on-cleaned MINUS raw-on-raw = {primary:+.4f}", flush=True)
    print(f"wrote {OUT_DIR}/final_summary.json + per_curve_predictions.csv", flush=True)


# ---- stage: prepare_cbv (build ONLY the CBV-cleaned arrays) ------------------
def stage_prepare_cbv(seed):
    """Reuse the existing raw experiment's rows/order/masks; build ONLY the four
    CBV-cleaned arrays with the frozen instrument JEPA + K=8 weight decoder +
    fixed area bases. Raw/direct-cleaned arrays, meta and manifest are untouched."""
    pm_path = os.path.join(PREP_DIR, "pretrain_meta.npz")
    em_path = os.path.join(PREP_DIR, "eval_meta.npz")
    if not (os.path.exists(pm_path) and os.path.exists(em_path)):
        raise RuntimeError("existing raw prepared arrays/meta missing; run the raw --stage prepare first")
    pm = np.load(pm_path, allow_pickle=True)
    em = np.load(em_path, allow_pickle=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    # --- eval cohort: byte-for-byte the same construction as stage_prepare ----
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next((c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id")
                     if c in phyts.columns), None)
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})
    eval_raw = _load_curves(EVAL_TGLC_PATH).rename(columns={"flux_raw": "aperture_flux"})
    matched, _ = match_phyts_tglc(phyts, eval_raw)
    matched = matched.rename(columns={"aperture_flux": "flux_raw"})
    if len(matched) != EXPECTED_N_EVAL:
        raise RuntimeError(f"eval cohort n={len(matched)} != {EXPECTED_N_EVAL}")
    eval_tics = matched["TIC"].to_numpy().astype(str)
    eval_gaia = matched["GAIADR3"].to_numpy()
    eval_sectors = matched["sector"].to_numpy()
    excl = set(zip(eval_gaia.tolist(), eval_sectors.tolist()))

    # --- pretraining rows: same removal + quality filter ---------------------
    pre = remove_eval_cohort(_load_curves(PRETRAIN_PATH), excl)
    pre_times, pre_fluxes, pre = _filter_curves(pre, MIN_GOOD)
    pre_tics = pre["TIC"].to_numpy().astype(str)

    # --- order/count MUST match the existing experiment ----------------------
    if not np.array_equal(pre_tics, pm["tics"].astype(str)):
        raise RuntimeError("pretrain TIC order differs from existing pretrain_meta.npz")
    if not np.array_equal(eval_tics, em["tics"].astype(str)):
        raise RuntimeError("eval TIC order differs from existing eval_meta.npz")
    if len(pre_tics) != EXPECTED_N_PRETRAIN:
        raise RuntimeError(f"pretrain count {len(pre_tics)} != {EXPECTED_N_PRETRAIN}")
    if len(eval_tics) != EXPECTED_N_EVAL:
        raise RuntimeError(f"eval count {len(eval_tics)} != {EXPECTED_N_EVAL}")

    # --- frozen instrument JEPA + K=8 weight decoder + fixed area bases -------
    inst = strict_load(FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                                  readout="mean", predictor_type="mlp").to(DEVICE),
                       torch.load(INST_CKPT, map_location=DEVICE)).eval()
    for p in inst.parameters():
        p.requires_grad_(False)
    wdec = strict_load(build_decoder(CBV_RANK).to(DEVICE),
                       torch.load(WEIGHT_DECODER_CKPT, map_location=DEVICE)).eval()
    for p in wdec.parameters():
        p.requires_grad_(False)
    bases = _load_bases(_resolve_bases_npz())
    before = (state_hash(inst.teacher), state_hash(inst.student),
              state_hash(inst.predictor), state_hash(wdec))
    print(f"CBV cleaning: {CBV_RANK} weights x {len(bases)} area bases "
          f"({_resolve_bases_npz()})", flush=True)

    # --- build ONLY the CBV arm (build_arms returns cbv in the 'cleaned' slot)-
    print("building CBV pretraining arm...", flush=True)
    _, _, pre_cbv_X, pre_cbv_M = build_arms(
        pre_times, pre_fluxes, inst, wdec, t0, t1, _areas_for(pre), bases)
    eval_times, eval_fluxes, matched2 = _filter_curves(matched, min_good=1)
    if len(matched2) != EXPECTED_N_EVAL:
        raise RuntimeError("an eval curve was dropped by the min-good filter")
    print("building CBV evaluation arm...", flush=True)
    _, _, ev_cbv_X, ev_cbv_M = build_arms(
        eval_times, eval_fluxes, inst, wdec, t0, t1, _areas_for(matched2), bases)

    after = (state_hash(inst.teacher), state_hash(inst.student),
             state_hash(inst.predictor), state_hash(wdec))
    if before != after:
        raise RuntimeError("instrument/weight-decoder changed during CBV preparation")

    # --- CBV masks must EXACTLY equal the existing raw masks ------------------
    if not np.array_equal(pre_cbv_M, np.load(os.path.join(PREP_DIR, "pretrain_raw_M.npy"))):
        raise RuntimeError("pretrain CBV masks != existing raw masks")
    if not np.array_equal(ev_cbv_M, np.load(os.path.join(PREP_DIR, "eval_raw_M.npy"))):
        raise RuntimeError("eval CBV masks != existing raw masks")

    np.save(os.path.join(PREP_DIR, "pretrain_cbv_X.npy"), pre_cbv_X)
    np.save(os.path.join(PREP_DIR, "pretrain_cbv_M.npy"), pre_cbv_M)
    np.save(os.path.join(PREP_DIR, "eval_cbv_X.npy"), ev_cbv_X)
    np.save(os.path.join(PREP_DIR, "eval_cbv_M.npy"), ev_cbv_M)
    print(f"wrote CBV arrays to {PREP_DIR} (pretrain {len(pre_cbv_X)}, eval {len(ev_cbv_X)})", flush=True)
    _ensure_init(seed)   # reuse (never overwrite) the existing seed-0 init


# ---- stage: evaluate_cbv (decisive matched comparison only) -----------------
def stage_evaluate_cbv(seed):
    """CBV-trained JEPA on CBV-cleaned curves vs existing raw-trained JEPA on raw
    curves. Same 1927/482 TIC-disjoint KNN split; nothing retrained."""
    manifest = _require_valid_prepared()
    em = np.load(os.path.join(PREP_DIR, "eval_meta.npz"), allow_pickle=True)
    y, tics, gaia, sectors = em["y"], em["tics"].astype(str), em["gaia"], em["sectors"]
    present = np.unique(y)

    for f in ("eval_cbv_X.npy", "eval_cbv_M.npy"):
        if not os.path.exists(os.path.join(PREP_DIR, f)):
            raise RuntimeError(f"missing {f}; run --stage prepare_cbv first")
    Xr = np.load(os.path.join(PREP_DIR, "eval_raw_X.npy"))
    Mr = np.load(os.path.join(PREP_DIR, "eval_raw_M.npy"))
    Xc = np.load(os.path.join(PREP_DIR, "eval_cbv_X.npy"))
    Mc = np.load(os.path.join(PREP_DIR, "eval_cbv_M.npy"))
    if not np.array_equal(Mr, Mc):
        raise RuntimeError("raw and CBV eval masks differ -- arms not matched")

    raw_ck = os.path.join(RAW_CKPT_DIR, f"physics_jepa_raw_s{seed}_best.pth")
    cbv_ck = os.path.join(PHYS_CKPT_DIR, f"physics_jepa_cbv_s{seed}_best.pth")

    def _load(ck):
        if not os.path.exists(ck):
            raise RuntimeError(f"missing checkpoint {ck}")
        m = build_latent_jepa().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    raw_jepa, cbv_jepa = _load(raw_ck), _load(cbv_ck)
    metas = {}
    for name, ck in (("raw", raw_ck), ("cbv", cbv_ck)):
        mp = ck.replace("_best.pth", "_meta.json")
        metas[name] = json.load(open(mp)) if os.path.exists(mp) else {}
    started_identical = (metas["raw"].get("init_hash") == metas["cbv"].get("init_hash")
                         and metas["raw"].get("init_hash") is not None)

    # same shared TIC-disjoint split as stage_evaluate (1927/482)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(len(tics)), y, groups=tics))

    lat_raw = encode_physics(raw_jepa, Xr, Mr)
    lat_cbv = encode_physics(cbv_jepa, Xc, Mc)
    raw_acc, raw_rec, _ = _classify(lat_raw, y, tr, te, present)
    cbv_acc, cbv_rec, _ = _classify(lat_cbv, y, tr, te, present)
    cbv_minus_raw = cbv_acc - raw_acc
    collapse = {
        "raw_jepa_on_raw": {"latent_std": float(lat_raw.std(0).mean()),
                            "effective_rank": _effective_rank(lat_raw)},
        "cbv_jepa_on_cbv": {"latent_std": float(lat_cbv.std(0).mean()),
                            "effective_rank": _effective_rank(lat_cbv)}}

    summary = {
        "comparison": "cbv_jepa_on_cbv MINUS raw_jepa_on_raw",
        "balanced_accuracy": {"raw_jepa_on_raw": raw_acc, "cbv_jepa_on_cbv": cbv_acc},
        "cbv_minus_raw": cbv_minus_raw,
        "per_class_recall": {"raw_jepa_on_raw": raw_rec, "cbv_jepa_on_cbv": cbv_rec},
        "collapse_metrics": collapse,
        "n_eval": int(len(tics)), "n_eval_train": int(len(tr)), "n_eval_test": int(len(te)),
        "classes": [CLASSES[c] for c in present], "seed": seed,
        "raw_ckpt": raw_ck, "cbv_ckpt": cbv_ck,
        "jepas_started_identical": bool(started_identical),
        "init_hash_raw": metas["raw"].get("init_hash"),
        "init_hash_cbv": metas["cbv"].get("init_hash"),
        "identical_eval_rows": True,
        "raw_cbv_eval_masks_identical": True,
        "eval_tic_sector_sha256": ordered_hash_tic_sector(tics, sectors),
        "eval_split_sha256": _ordered_hash(np.concatenate([tr, te])),
        "weight_decoder_ckpt": WEIGHT_DECODER_CKPT, "bases_npz": _resolve_bases_npz(),
        "phyts_flux_used": False, "git_commit": _git_commit(),
    }
    out_path = os.path.join(OUT_DIR, "cbv_final_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"raw JEPA on raw : {raw_acc:.4f}", flush=True)
    print(f"cbv JEPA on cbv : {cbv_acc:.4f}", flush=True)
    print(f"CBV - RAW       : {cbv_minus_raw:+.4f}   (started_identical={started_identical})", flush=True)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prepare", "train", "evaluate", "prepare_cbv", "evaluate_cbv"])
    ap.add_argument("--arm", choices=["raw", "cleaned", "cbv"], default=os.environ.get("ARM"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PHYS_CKPT_DIR, exist_ok=True)
    print(f"CLEAN_MODE={CLEAN_MODE} OUT_DIR={OUT_DIR} PHYS_CKPT_DIR={PHYS_CKPT_DIR} "
          f"RAW_CKPT_DIR={RAW_CKPT_DIR}", flush=True)
    if CLEAN_MODE == "cbv":
        print(f"  WEIGHT_DECODER_CKPT={WEIGHT_DECODER_CKPT}", flush=True)
    elif AREA_HEADS:
        print(f"  AREA_DECODER_CKPT={AREA_DECODER_CKPT}", flush=True)
    elif LEARNED:
        print(f"  LEARNED_CKPT={LEARNED_CKPT}", flush=True)

    if args.stage == "prepare":
        stage_prepare(args.seed)
    elif args.stage == "prepare_cbv":
        stage_prepare_cbv(args.seed)
    elif args.stage == "train":
        arm = args.arm
        if arm is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            arm = {"0": "raw", "1": "cleaned"}[os.environ["SLURM_ARRAY_TASK_ID"]]
        if arm is None:
            raise SystemExit("train stage needs --arm raw|cleaned|cbv (or array task 0/1)")
        stage_train(arm, args.seed)
    elif args.stage == "evaluate_cbv":
        stage_evaluate_cbv(args.seed)
    else:
        stage_evaluate(args.seed)


if __name__ == "__main__":
    main()
