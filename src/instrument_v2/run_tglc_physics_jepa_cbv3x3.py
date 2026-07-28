from __future__ import annotations
"""Matched 3x3 physics-JEPA pretraining comparison on raw TGLC (Sector 14):
raw vs DIRECT-cleaned vs CBV-WEIGHT-cleaned curves.

The raw and direct-cleaned arms are NOT rebuilt or retrained here. Their
prepared arrays, per-seed init and best checkpoints are REUSED from the frozen
tglc_physics_jepa_ab experiment; any staleness (manifest mismatch, missing
files, changed init) HARD-FAILS instead of silently mixing experiments.

This module adds exactly one new arm: each curve is cleaned by the frozen K=8
CBV-weight decoder

    z_instrument -> MLP -> w in R^8,   template = B_area @ w

with the FIXED area bases built from instrument-training TICs only
(regional_cbv cache). The template is subtracted at NATIVE timestamps in flux
units (same rule as the direct arm: no scale fit), then the physics
preprocessing runs exactly once. Missing area bases hard-fail; a curve with
<8 valid instrument-grid bins falls back to raw exactly like the direct arm
and is COUNTED.

A third physics JEPA is then trained on the CBV-cleaned curves through the
SAME train_arm_from_arrays as the raw/direct arms (identical init, seed,
train/val indices, batch order, segment-mask seeds, architecture, optimizer,
30 epochs, best-val checkpoint rule) and the full 3x3 KNN matrix is evaluated
on identical labeled rows and indices.

Stages:
  --stage prepare    validate + reuse the ab experiment; build CBV-cleaned arrays
  --stage train      pretrain the cbv-arm physics JEPA
  --stage evaluate   frozen 3x3 KNN probe + collapse metrics

Primary result = (cbv JEPA on cbv-cleaned) - (raw JEPA on raw).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from sklearn.model_selection import GroupShuffleSplit

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder, sha256_file
from src.instrument_v2.regional_cbv import load_area_bases
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.eval_phyts_instrument_ab import DEVICE, GRID, physics_grid, encode_physics
from src.instrument_v2.eval_phyts_raw_tglc_ab import (
    PHYTS_PATH, INST_CKPT, GRID_RANGE, match_phyts_tglc, ordered_hash_tic_sector,
)
from src.instrument_v2.run_tglc_physics_jepa_ab import (
    PRETRAIN_PATH, EVAL_TGLC_PATH, EXPECTED_TIC_SECTOR_SHA, EXPECTED_N_EVAL, MIN_GOOD,
    CONFIG_KEYS as AB_CONFIG_KEYS,
    PREP_DIR as AB_PREP_DIR,
    PHYS_CKPT_DIR as AB_CKPT_DIR,
    current_config as ab_current_config,
    _require_valid_prepared as ab_require_valid_prepared,
    _file_sig, _ordered_hash, _git_commit, _grid_times,
    grid_for_instrument, subtract_native, remove_eval_cohort, _filter_curves,
    strict_load, train_arm_from_arrays, _classify, _effective_rank,
)

# ---- paths / config ---------------------------------------------------------
CBV_RANK = 8                                   # fixed for this controlled comparison
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_qclean_v1"))
WEIGHT_DECODER_CKPT = os.environ.get(
    "WEIGHT_DECODER_CKPT",
    os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
CBV_BASES_NPZ = os.environ.get("CBV_BASES_NPZ", "")     # empty -> unique glob below
OUT_DIR = os.environ.get(
    "CBV3X3_OUT_DIR", os.path.join("artifacts", "instrument_v2", "tglc_physics_jepa_cbv3x3"))
PREP_DIR = os.path.join(OUT_DIR, "prepared")
PHYS_CKPT_DIR = os.environ.get(
    "CBV3X3_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/tglc_physics_jepa_cbv3x3")

# every ab key must STILL match (raw/direct arrays and init are reused), plus
# the cbv-specific artifacts
CBV_CONFIG_KEYS = AB_CONFIG_KEYS + ["weight_decoder_sig", "bases_sig", "cbv_rank"]

ARMS = ("raw", "direct", "cbv")
CELLS = {f"{jm}_jepa_on_{dm}": (jm, dm) for jm in ARMS for dm in ARMS}


# ---- config / manifest ------------------------------------------------------
def resolve_bases_npz():
    """The one cached train-TIC area-basis npz. Ambiguity or absence hard-fails."""
    if CBV_BASES_NPZ:
        if not os.path.exists(CBV_BASES_NPZ):
            raise RuntimeError(f"CBV_BASES_NPZ does not exist: {CBV_BASES_NPZ}")
        return CBV_BASES_NPZ
    pattern = os.path.join(
        GROUP_ART_DIR, f"area_group_cbv_r{CBV_RANK}_g32_mv16_q16437_tglc0_*.npz")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"need exactly one basis npz matching {pattern}, "
                           f"found {len(hits)}: {hits} -- set CBV_BASES_NPZ explicitly")
    return hits[0]


def current_config():
    bases_npz = resolve_bases_npz()
    cfg = dict(ab_current_config())
    cfg.update({"weight_decoder_ckpt": WEIGHT_DECODER_CKPT,
                "weight_decoder_sig": _file_sig(WEIGHT_DECODER_CKPT),
                "bases_npz": bases_npz, "bases_sig": _file_sig(bases_npz),
                "cbv_rank": CBV_RANK})
    return cfg


def assert_cbv_manifest_matches(manifest, current):
    """Refuse stale prepared CBV data: every ab key AND every cbv key must match."""
    for k in CBV_CONFIG_KEYS:
        if manifest.get(k) != current.get(k):
            raise RuntimeError(f"stale cbv prepared data: '{k}' changed "
                               f"({manifest.get(k)} != {current.get(k)}) -- re-run --stage prepare")


def require_shared_init(init_path, expected_sha=None):
    """The per-seed init the raw/direct JEPAs trained from. Creating a new one
    here would break 'same initialization', so a missing/changed file hard-fails."""
    if not os.path.exists(init_path):
        raise RuntimeError(f"missing shared physics-JEPA init {init_path}; the raw/direct "
                           "arms trained from it -- refusing to create a new one")
    sha = sha256_file(init_path)
    if expected_sha is not None and sha != expected_sha:
        raise RuntimeError(f"shared init {init_path} changed since prepare "
                           f"({sha[:12]} != {expected_sha[:12]})")
    return sha


def _require_valid_prepared_cbv():
    """Both manifests (frozen ab experiment AND this cbv extension) must be fresh."""
    ab_manifest = ab_require_valid_prepared()
    manifest_path = os.path.join(PREP_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError("no cbv prepared data; run --stage prepare first")
    manifest = json.load(open(manifest_path))
    assert_cbv_manifest_matches(manifest, current_config())
    return ab_manifest, manifest


# ---- data loading with area assignment --------------------------------------
def _load_curves_area(path):
    """run_tglc_physics_jepa_ab._load_curves plus an 'area' column (same rows,
    same order). dense_v2 already stores the merged area; other parquets must
    carry camera/ccd/ra/dec so the EXISTING src.regions.areas.add_area
    assignment applies. No silent row drops here -- invalid areas hard-fail
    later in require_valid_areas."""
    cols = set(pq.read_schema(path).names)
    flux_col = "aperture_flux" if "aperture_flux" in cols else "flux"
    want = ["TIC", "sector", "GAIADR3", "time", flux_col, "TESS_flags", "TGLC_flags"]
    have_area = "area" in cols
    if have_area:
        want.append("area")
    elif {"camera", "ccd", "ra", "dec"} <= cols:
        want += ["camera", "ccd", "ra", "dec"]
    else:
        raise RuntimeError(f"{path} has neither 'area' nor camera/ccd/ra/dec -- "
                           "cannot assign area-specific CBV bases")
    df = pd.read_parquet(path, columns=want)
    df["TIC"] = df["TIC"].astype(str)
    df = df.rename(columns={flux_col: "flux_raw"})
    if not have_area:
        from src.regions.areas import add_area          # needs tess_stars2px (cluster)
        df = add_area(df)
    return df


def require_valid_areas(df, name):
    """Every kept row must have a real area; -1/NaN would silently get the wrong
    basis, so it is a hard error. Returns the int area array."""
    a = pd.to_numeric(df["area"], errors="coerce")
    bad = a.isna() | (a == -1)
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())}/{len(df)} {name} rows lack a valid area "
                           "-- refusing to clean without the correct basis")
    return a.astype(int).to_numpy()


# ---- CBV-weight cleaning ----------------------------------------------------
def area_basis(bases, area):
    """The curve's own area-specific (1024, 8) basis. A missing basis HARD-FAILS
    -- silently returning the raw curve would poison the matched comparison."""
    a = int(area)
    if a not in bases:
        raise RuntimeError(f"no CBV basis for area {a} -- refusing to fall back to raw")
    B = np.asarray(bases[a], dtype=np.float64)
    if B.shape != (GRID, CBV_RANK):
        raise RuntimeError(f"area {a} basis shape {B.shape} != ({GRID}, {CBV_RANK})")
    return B


def batched_weights(inst, wdecoder, Xg, Mg, batch=512):
    """One batched GPU pass: frozen instrument JEPA (view='predicted') then the
    frozen weight decoder. Returns (n, CBV_RANK) weights."""
    W = np.zeros((len(Xg), CBV_RANK), np.float32)
    with torch.no_grad():
        for s in range(0, len(Xg), batch):
            f = torch.tensor(Xg[s:s + batch], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mg[s:s + batch], dtype=torch.float32, device=DEVICE)
            w = wdecoder(inst.encode(f, m, view="predicted"))
            if w.shape[1] != CBV_RANK:
                raise RuntimeError(f"weight decoder emitted {w.shape[1]} != {CBV_RANK} weights")
            W[s:s + batch] = w.detach().cpu().numpy()
    return W


def build_cbv_arm(times, fluxes, areas, inst, wdecoder, bases, t0, t1):
    """Matched CBV-cleaned physics inputs. Per curve: median/MAD normalize onto
    the shared S14 grid, weight-decode once (batched), template = B_area @ w,
    subtract at NATIVE timestamps in flux units, then physics preprocessing
    exactly once. Also recomputes the raw arm from the same filtered curves so
    the caller can assert bit-identity with the frozen ab arrays.

    Returns (raw_X, raw_M, cbv_X, cbv_M, n_fallback). n_fallback counts curves
    with <8 valid instrument-grid bins that stay raw -- the SAME rule as the
    direct arm. Missing bases raise before any fallback."""
    n = len(times)
    grid_times = _grid_times(t0, t1)
    Xg = np.zeros((n, GRID), np.float32); Mg = np.zeros((n, GRID), np.float32)
    scale = np.zeros(n); valids = []
    for i in range(n):
        Xg[i], Mg[i], _, scale[i], v = grid_for_instrument(times[i], fluxes[i], t0, t1)
        valids.append(v)
    W = batched_weights(inst, wdecoder, Xg, Mg)

    raw_X = np.zeros((n, GRID), np.float32); raw_M = np.zeros((n, GRID), np.float32)
    cbv_X = np.zeros((n, GRID), np.float32); cbv_M = np.zeros((n, GRID), np.float32)
    n_fallback = 0
    for i in range(n):
        ft, ff = times[i], fluxes[i]
        raw_X[i], raw_M[i] = physics_grid(ft, ff)
        B = area_basis(bases, areas[i])              # hard-fails on a missing basis
        if valids[i].sum() >= 8:
            template = B @ W[i].astype(np.float64)   # (1024,) = B_area @ w
        else:
            template = None                          # same fallback rule as the direct arm
            n_fallback += 1
        cbv_X[i], cbv_M[i] = physics_grid(
            ft, subtract_native(ft, ff, template, valids[i], scale[i], grid_times))
        if i % 5000 == 0:
            print(f"    build_cbv_arm {i}/{n}", flush=True)
    assert np.array_equal(raw_M, cbv_M), "raw and cbv masks differ"
    return raw_X, raw_M, cbv_X, cbv_M, n_fallback


def _area_counts(areas):
    vals, counts = np.unique(np.asarray(areas, dtype=int), return_counts=True)
    return {int(a): int(c) for a, c in zip(vals, counts)}


# ---- stage: prepare ---------------------------------------------------------
def stage_prepare(seed):
    os.makedirs(PREP_DIR, exist_ok=True)

    # --- frozen ab (raw/direct) experiment must exist and be fresh -----------
    ab_manifest = ab_require_valid_prepared()
    init_path = os.path.join(AB_CKPT_DIR, f"physics_jepa_init_s{seed}.pth")
    init_sha = require_shared_init(init_path)
    cfg = current_config()
    bases_npz = cfg["bases_npz"]

    manifest_path = os.path.join(PREP_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        saved = json.load(open(manifest_path))
        try:
            assert_cbv_manifest_matches(saved, cfg)
            print("cbv prepared data already valid; nothing to do", flush=True)
            return
        except RuntimeError as e:
            print(f"cbv manifest mismatch ({e}); rebuilding", flush=True)

    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    # --- eval cohort: SAME construction and hash checks as the ab experiment --
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14].reset_index(drop=True)
    phyts["TIC"] = phyts["TIC"].astype(str)
    gaia_col = next((c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id")
                     if c in phyts.columns), None)
    phyts = phyts[["TIC", "sector", "label", gaia_col]].rename(columns={gaia_col: "phyts_gaia"})
    eval_raw = _load_curves_area(EVAL_TGLC_PATH).rename(columns={"flux_raw": "aperture_flux"})
    matched, _ = match_phyts_tglc(phyts, eval_raw)
    matched = matched.rename(columns={"aperture_flux": "flux_raw"})
    if len(matched) != EXPECTED_N_EVAL:
        raise RuntimeError(f"eval cohort n={len(matched)} != {EXPECTED_N_EVAL}")
    eval_tics = matched["TIC"].to_numpy().astype(str)
    eval_sectors = matched["sector"].to_numpy()
    eval_gaia = matched["GAIADR3"].to_numpy()
    if ordered_hash_tic_sector(eval_tics, eval_sectors) != EXPECTED_TIC_SECTOR_SHA:
        raise RuntimeError("eval cohort TIC/sector hash != prior frozen experiment")
    if _ordered_hash(eval_gaia, eval_tics) != ab_manifest["eval_gaia_tic_sha256"]:
        raise RuntimeError("eval cohort rows differ from the frozen ab experiment")
    eval_y = np.array([CLASS_TO_IDX[l] for l in matched["label"]], dtype=np.int64)
    if len(np.unique(eval_y)) != 7:
        raise RuntimeError("eval cohort missing a class")
    excl = set(zip(eval_gaia.tolist(), eval_sectors.tolist()))

    # --- pretraining rows: same removal + filter, then identity vs ab --------
    pre = _load_curves_area(PRETRAIN_PATH)
    pre = remove_eval_cohort(pre, excl)                    # hard-fails if any eval star remains
    pre_times, pre_fluxes, pre = _filter_curves(pre, MIN_GOOD)
    pre_tics = pre["TIC"].to_numpy().astype(str)
    if _ordered_hash(pre["GAIADR3"].to_numpy(), pre_tics) != ab_manifest["pretrain_gaia_tic_sha256"]:
        raise RuntimeError("pretraining rows differ from the frozen ab experiment")
    old_meta = np.load(os.path.join(AB_PREP_DIR, "pretrain_meta.npz"), allow_pickle=True)
    if not np.array_equal(old_meta["tics"].astype(str), pre_tics):
        raise RuntimeError("pretraining TIC order differs from the frozen ab experiment")

    pre_areas = require_valid_areas(pre, "pretrain")
    eval_times, eval_fluxes, matched2 = _filter_curves(matched, min_good=1)
    if len(matched2) != EXPECTED_N_EVAL:
        raise RuntimeError("an eval curve was dropped by the min-good filter")
    eval_areas = require_valid_areas(matched2, "eval")

    # --- frozen instrument JEPA + K=8 weight decoder + train-only bases ------
    inst = strict_load(FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                                  n_layers=4, readout="mean",
                                                  predictor_type="mlp").to(DEVICE),
                       torch.load(INST_CKPT, map_location=DEVICE)).eval()
    for p in inst.parameters():
        p.requires_grad_(False)
    wdecoder = strict_load(build_decoder(CBV_RANK).to(DEVICE),
                           torch.load(WEIGHT_DECODER_CKPT, map_location=DEVICE)).eval()
    for p in wdecoder.parameters():
        p.requires_grad_(False)
    bases = load_area_bases(bases_npz)
    for a in bases:
        area_basis(bases, a)                              # validate every shape up front
    inst_before = (state_hash(inst.teacher), state_hash(inst.student),
                   state_hash(inst.predictor), state_hash(wdecoder))
    print(f"frozen instrument {INST_CKPT}", flush=True)
    print(f"frozen weight decoder {WEIGHT_DECODER_CKPT}", flush=True)
    print(f"model hashes: teacher {inst_before[0][:12]} student {inst_before[1][:12]} "
          f"predictor {inst_before[2][:12]} weight_decoder {inst_before[3][:12]}", flush=True)
    print(f"bases: {bases_npz} ({len(bases)} areas)", flush=True)

    # --- build the CBV arm (+ raw recomputation for the identity check) ------
    print("building CBV pretraining arm...", flush=True)
    pre_raw_X, pre_raw_M, pre_cbv_X, pre_cbv_M, pre_fb = build_cbv_arm(
        pre_times, pre_fluxes, pre_areas, inst, wdecoder, bases, t0, t1)
    print("building CBV evaluation arm...", flush=True)
    ev_raw_X, ev_raw_M, ev_cbv_X, ev_cbv_M, ev_fb = build_cbv_arm(
        eval_times, eval_fluxes, eval_areas, inst, wdecoder, bases, t0, t1)

    inst_after = (state_hash(inst.teacher), state_hash(inst.student),
                  state_hash(inst.predictor), state_hash(wdecoder))
    if inst_before != inst_after:
        raise RuntimeError("instrument/weight-decoder changed during preparation")

    # --- identity with the frozen ab arrays: same stars, timestamps, ordering,
    #     masks. Any drift hard-fails. ----------------------------------------
    for name, mine in (("pretrain_raw_X", pre_raw_X), ("pretrain_raw_M", pre_raw_M),
                       ("eval_raw_X", ev_raw_X), ("eval_raw_M", ev_raw_M)):
        old = np.load(os.path.join(AB_PREP_DIR, f"{name}.npy"))
        if not np.array_equal(old, mine):
            raise RuntimeError(f"{name} differs from the frozen ab experiment -- "
                               "the raw/direct/cbv arms would not be matched")
    for name, mine in (("pretrain_cleaned_M", pre_cbv_M), ("eval_cleaned_M", ev_cbv_M)):
        old = np.load(os.path.join(AB_PREP_DIR, f"{name}.npy"))
        if not np.array_equal(old, mine):
            raise RuntimeError(f"{name} mask differs from the cbv mask -- arms not matched")

    np.save(os.path.join(PREP_DIR, "pretrain_cbv_X.npy"), pre_cbv_X)
    np.save(os.path.join(PREP_DIR, "pretrain_cbv_M.npy"), pre_cbv_M)
    np.save(os.path.join(PREP_DIR, "eval_cbv_X.npy"), ev_cbv_X)
    np.save(os.path.join(PREP_DIR, "eval_cbv_M.npy"), ev_cbv_M)

    bases_json = os.path.splitext(bases_npz)[0] + ".json"
    hashes = {"inst_ckpt": sha256_file(INST_CKPT),
              "weight_decoder_ckpt": sha256_file(WEIGHT_DECODER_CKPT),
              "direct_decoder_ckpt": sha256_file(cfg["decoder_ckpt"]),
              "bases_npz": sha256_file(bases_npz),
              "physics_jepa_init": init_sha}
    for name in ("pretrain_cbv_X", "pretrain_cbv_M", "eval_cbv_X", "eval_cbv_M"):
        hashes[name] = sha256_file(os.path.join(PREP_DIR, f"{name}.npy"))
    for name in ("pretrain_raw_X", "pretrain_raw_M", "pretrain_cleaned_X", "pretrain_cleaned_M",
                 "eval_raw_X", "eval_raw_M", "eval_cleaned_X", "eval_cleaned_M",
                 "pretrain_meta", "eval_meta"):
        ext = ".npz" if name.endswith("meta") else ".npy"
        hashes[f"ab_{name}"] = sha256_file(os.path.join(AB_PREP_DIR, name + ext))

    manifest = {**cfg,
                "git_commit": _git_commit(),
                "seed": seed,
                "n_pretrain": int(len(pre)), "n_eval": int(EXPECTED_N_EVAL),
                "pretrain_gaia_tic_sha256": ab_manifest["pretrain_gaia_tic_sha256"],
                "eval_gaia_tic_sha256": ab_manifest["eval_gaia_tic_sha256"],
                "eval_tic_sector_sha256": ab_manifest["eval_tic_sector_sha256"],
                "exclusion_hash": ab_manifest["exclusion_hash"],
                "curves_per_area": {"pretrain": _area_counts(pre_areas),
                                    "eval": _area_counts(eval_areas)},
                "n_pretrain_fallback": int(pre_fb), "n_eval_fallback": int(ev_fb),
                "n_missing_basis": 0,               # missing bases raise; reaching here means none
                "init_sha256": init_sha,
                "model_hashes": {"inst_teacher": inst_before[0], "inst_student": inst_before[1],
                                 "inst_predictor": inst_before[2], "weight_decoder": inst_before[3]},
                "sha256": hashes,
                "bases_provenance": (json.load(open(bases_json))
                                     if os.path.exists(bases_json) else None),
                "raw_arrays_identical_to_ab": True,
                "masks_identical_all_arms": True,
                "instrument_decoder_unchanged": True}
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote CBV arrays + manifest to {PREP_DIR} "
          f"(fallbacks: pretrain {pre_fb}, eval {ev_fb})", flush=True)


# ---- stage: train -----------------------------------------------------------
def stage_train(seed):
    _, manifest = _require_valid_prepared_cbv()
    meta = np.load(os.path.join(AB_PREP_DIR, "pretrain_meta.npz"))   # SAME indices as raw/direct
    X = torch.from_numpy(np.load(os.path.join(PREP_DIR, "pretrain_cbv_X.npy")))
    M = torch.from_numpy(np.load(os.path.join(PREP_DIR, "pretrain_cbv_M.npy")))
    init_path = os.path.join(AB_CKPT_DIR, f"physics_jepa_init_s{seed}.pth")
    require_shared_init(init_path, expected_sha=manifest["init_sha256"])
    os.makedirs(PHYS_CKPT_DIR, exist_ok=True)
    train_arm_from_arrays("cbv", seed, X, M, meta["train_idx"], meta["val_idx"],
                          init_path, PHYS_CKPT_DIR)


# ---- stage: evaluate --------------------------------------------------------
def _arm_ckpt(arm, seed):
    """raw/direct checkpoints are REUSED from the frozen ab experiment (the
    direct arm was named 'cleaned' there); only cbv lives in this experiment."""
    if arm == "cbv":
        return os.path.join(PHYS_CKPT_DIR, f"physics_jepa_cbv_s{seed}_best.pth")
    ab_name = {"raw": "raw", "direct": "cleaned"}[arm]
    return os.path.join(AB_CKPT_DIR, f"physics_jepa_{ab_name}_s{seed}_best.pth")


def stage_evaluate(seed):
    ab_manifest, manifest = _require_valid_prepared_cbv()
    em = np.load(os.path.join(AB_PREP_DIR, "eval_meta.npz"), allow_pickle=True)
    y, tics, gaia, sectors = em["y"], em["tics"].astype(str), em["gaia"], em["sectors"]
    present = np.unique(y)

    X = {"raw": np.load(os.path.join(AB_PREP_DIR, "eval_raw_X.npy")),
         "direct": np.load(os.path.join(AB_PREP_DIR, "eval_cleaned_X.npy")),
         "cbv": np.load(os.path.join(PREP_DIR, "eval_cbv_X.npy"))}
    Mk = {"raw": np.load(os.path.join(AB_PREP_DIR, "eval_raw_M.npy")),
          "direct": np.load(os.path.join(AB_PREP_DIR, "eval_cleaned_M.npy")),
          "cbv": np.load(os.path.join(PREP_DIR, "eval_cbv_M.npy"))}
    if not (np.array_equal(Mk["raw"], Mk["direct"]) and np.array_equal(Mk["raw"], Mk["cbv"])):
        raise RuntimeError("raw/direct/cbv evaluation masks differ -- arms not matched")

    jepas, metas = {}, {}
    for arm in ARMS:
        ck = _arm_ckpt(arm, seed)
        if not os.path.exists(ck):
            raise RuntimeError(f"missing checkpoint {ck}; train the {arm} arm first")
        m = build_latent_jepa().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        jepas[arm] = m
        mp = os.path.splitext(ck)[0].replace("_best", "_meta") + ".json"
        if not os.path.exists(mp):
            raise RuntimeError(f"missing training meta {mp} -- cannot verify shared init")
        metas[arm] = json.load(open(mp))
    init_hashes = {a: metas[a].get("init_hash") for a in ARMS}
    if None in init_hashes.values() or len(set(init_hashes.values())) != 1:
        raise RuntimeError(f"physics JEPAs did NOT start from one init: {init_hashes}")
    for key in ("epochs", "batch_size", "lr", "var_weight", "config"):
        vals = {a: metas[a].get(key) for a in ARMS}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) != 1:
            raise RuntimeError(f"training '{key}' differs across arms: {vals} -- "
                               "refusing to mix experiments")

    # one shared TIC-disjoint split; all NINE cells use identical rows/indices
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(len(tics)), y, groups=tics))

    acc, recall, preds, collapse = {}, {}, {}, {}
    for name, (jm, dm) in CELLS.items():
        lat = encode_physics(jepas[jm], X[dm], Mk[dm])
        acc[name], recall[name], preds[name] = _classify(lat, y, tr, te, present)
        collapse[name] = {"latent_std": float(lat.std(0).mean()),
                          "effective_rank": _effective_rank(lat)}

    primary = acc["cbv_jepa_on_cbv"] - acc["raw_jepa_on_raw"]
    vs_direct = acc["cbv_jepa_on_cbv"] - acc["direct_jepa_on_direct"]

    rows = []
    for j, gi in enumerate(te):
        r = {"TIC": tics[gi], "GaiaDR3": gaia[gi], "true_label": CLASSES[y[gi]], "split": "test"}
        for name in CELLS:
            r[f"pred_{name}"] = CLASSES[preds[name][j]]
        rows.append(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_curve_predictions.csv"), index=False)

    # decoder validation metrics are REQUIRED in this report
    wd_summary_path = os.path.join(os.path.dirname(WEIGHT_DECODER_CKPT), "final_summary.json")
    if not os.path.exists(wd_summary_path):
        raise RuntimeError(f"missing weight-decoder summary {wd_summary_path}; "
                           "run the DECODER_MODE=weights decode job first")
    wd_metrics = json.load(open(wd_summary_path)).get("metrics")

    ckpt_hashes = {f"{arm}_jepa_ckpt": sha256_file(_arm_ckpt(arm, seed)) for arm in ARMS}
    summary = {
        "balanced_accuracy": acc,
        "primary_cbv_on_cbv_minus_raw_on_raw": primary,
        "cbv_on_cbv_minus_direct_on_direct": vs_direct,
        "per_class_recall": recall,
        "collapse_metrics": collapse,
        "weight_decoder_validation": wd_metrics,
        "curves_per_area": manifest["curves_per_area"],
        "cleaning_fallbacks": {"pretrain": manifest["n_pretrain_fallback"],
                               "eval": manifest["n_eval_fallback"],
                               "missing_basis": manifest["n_missing_basis"]},
        "n_pretrain": manifest["n_pretrain"],
        "n_pretrain_train": ab_manifest["n_pretrain_train"],
        "n_pretrain_val": ab_manifest["n_pretrain_val"],
        "n_eval": int(len(tics)), "n_eval_train": int(len(tr)), "n_eval_test": int(len(te)),
        "classes": [CLASSES[c] for c in present],
        "seed": seed,
        "ckpts": {arm: _arm_ckpt(arm, seed) for arm in ARMS},
        "pretrain_val_loss": {a: metas[a].get("val_loss") for a in ARMS},
        "jepas_started_identical": True,
        "init_hash": init_hashes["cbv"],
        "sha256": {**manifest["sha256"], **ckpt_hashes},
        "paths": {"pretrain": PRETRAIN_PATH, "eval_tglc": EVAL_TGLC_PATH, "phyts": PHYTS_PATH,
                  "inst_ckpt": INST_CKPT, "weight_decoder_ckpt": WEIGHT_DECODER_CKPT,
                  "direct_decoder_ckpt": manifest["decoder_ckpt"],
                  "bases_npz": manifest["bases_npz"], "grid_range": GRID_RANGE},
        "eval_tic_sector_sha256": ordered_hash_tic_sector(tics, sectors),
        "eval_split_sha256": _ordered_hash(np.concatenate([tr, te])),
        "exclusion_hash": ab_manifest["exclusion_hash"],
        "eval_excluded_from_pretraining": True,
        "masks_identical_all_arms": True,
        "phyts_flux_used": False,
        "git_commit": _git_commit(),
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("3x3 balanced accuracy (rows: pretraining arm; cols: eval curves):", flush=True)
    print(f"{'':>14s}  {'raw':>8s} {'direct':>8s} {'cbv':>8s}", flush=True)
    for jm in ARMS:
        cells = " ".join(f"{acc[f'{jm}_jepa_on_{dm}']:>8.4f}" for dm in ARMS)
        print(f"  {jm + ' JEPA':>12s}  {cells}", flush=True)
    print(f"PRIMARY cbv-on-cbv MINUS raw-on-raw     = {primary:+.4f}", flush=True)
    print(f"        cbv-on-cbv MINUS direct-on-direct = {vs_direct:+.4f}", flush=True)
    print(f"wrote {OUT_DIR}/final_summary.json + per_curve_predictions.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "train", "evaluate"])
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("================ resolved configuration ================", flush=True)
    print(f"  git commit          : {_git_commit()}", flush=True)
    print(f"  stage / seed        : {args.stage} / {args.seed}", flush=True)
    print(f"  PRETRAIN_PATH       : {PRETRAIN_PATH}", flush=True)
    print(f"  EVAL_TGLC_PATH      : {EVAL_TGLC_PATH}", flush=True)
    print(f"  INST_CKPT           : {INST_CKPT}", flush=True)
    print(f"  WEIGHT_DECODER_CKPT : {WEIGHT_DECODER_CKPT}", flush=True)
    print(f"  CBV_BASES_NPZ       : {CBV_BASES_NPZ or '(unique glob in ' + GROUP_ART_DIR + ')'}", flush=True)
    print(f"  AB prep / ckpts     : {AB_PREP_DIR} | {AB_CKPT_DIR}", flush=True)
    print(f"  CBV prep / ckpts    : {PREP_DIR} | {PHYS_CKPT_DIR}", flush=True)
    print("========================================================", flush=True)

    if args.stage == "prepare":
        stage_prepare(args.seed)
    elif args.stage == "train":
        stage_train(args.seed)
    else:
        stage_evaluate(args.seed)


if __name__ == "__main__":
    main()
