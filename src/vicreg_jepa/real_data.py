"""Real TGLC light curves -> the arrays the VICReg-JEPA trains on.

Data contract (matches src/instrument_v2/sector14_dataset.py so that results
here stay comparable with the existing baselines):
  - RAW `flux` only. `flux_cal` is never read.
  - A cadence is dropped when (TESS_flags & 16437) != 0, or TGLC_flags != 0,
    or the flux is not finite. Dropped cadences stay MISSING -- never infilled.
  - Median/MAD normalisation computed from the surviving cadences only.
  - Shared per-sector grid: one global time range per sector, 1024 equal bins,
    identical for every star. Several cadences in a bin are averaged; a bin is
    observed if any cadence lands in it. Unobserved bins hold 0.0 and mask 0.
  - Region id = camera*100 + ccd*10 + ring, ring in 1..4 by angular distance
    from the camera boresight (src/regions/areas.py convention).

test_real_data.py asserts the local helpers are bit-identical to the repo's
reference implementations, so the model package stays import-standalone while
the conventions cannot silently drift.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict

import numpy as np

BAD_TESS_MASK = 16437          # 1 | 4 | 16 | 32 | 16384
GRID_1024 = 1024
RADIUS_CAMERA = np.sqrt(12.0 ** 2 + 12.0 ** 2)

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}


@dataclass
class DataConfig:
    parquet_glob: str = "disentangle_attempt/data_multichip/cross_sector_raw.parquet/*.parquet"
    out: str = "artifacts/vicreg_jepa/curves.npz"
    seq_len: int = GRID_1024
    group_size: int = 32
    min_observed_frac: float = 0.25   # curve must keep >=25% of its bins
    val_frac: float = 0.15
    test_frac: float = 0.20
    split_seed: int = 43
    label_csv: str = ""               # optional: TIC,label  (PhyTS)
    require_labels: bool = False
    strict_quality: bool = False      # True = drop on ANY nonzero TESS bit


# ------------------------------------------------------------------ helpers
def quality_keep(tess_flags, tglc_flags, flux, strict=False):
    """Cadences whose every quality gate passes and whose flux is finite.

    strict=False (default): drop only TESS_flags & 16437 (attitude tweak, coarse
    point, desat, ...). Stray-light cadences (bit 2048) are KEPT.
    strict=True: drop on ANY nonzero TESS bit, stray light included.
    """
    tess = np.asarray(tess_flags).astype(np.int64)
    tglc = np.asarray(tglc_flags).astype(np.int64)
    f = np.asarray(flux, dtype=np.float64)
    bad_tess = (tess != 0) if strict else ((tess & BAD_TESS_MASK) != 0)
    return (~bad_tess) & (tglc == 0) & np.isfinite(f)


def normalize_median_mad(flux):
    """Robust z-score from OBSERVED flux only. Mirrors the repo helper."""
    flux = np.asarray(flux, dtype=np.float64)
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = 1.0
    return (flux - med) / scale, med, mad


def build_shared_sector(times, fluxes, grid_length=GRID_1024, bounds=None):
    """One global time range -> grid_length bins, identical for every star."""
    if bounds is None:
        t0 = min(float(np.min(t)) for t in times)
        t1 = max(float(np.max(t)) for t in times)
    else:
        t0, t1 = bounds
    span = max(t1 - t0, 1e-9)
    X = np.zeros((len(fluxes), grid_length), dtype=np.float32)
    M = np.zeros_like(X)
    for i, (t, f) in enumerate(zip(times, fluxes)):
        b = np.clip(((np.asarray(t) - t0) / span * grid_length).astype(np.int64),
                    0, grid_length - 1)
        total = np.zeros(grid_length)
        count = np.zeros(grid_length)
        np.add.at(total, b, np.asarray(f, dtype=np.float64))
        np.add.at(count, b, 1.0)
        hit = count > 0
        X[i, hit] = (total[hit] / count[hit]).astype(np.float32)
        M[i, hit] = 1.0
    return X, M, (t0, t1)


def sphere_distance(ra_1, dec_1, ra_2, dec_2):
    ra_1, dec_1, ra_2, dec_2 = map(np.radians, (ra_1, dec_1, ra_2, dec_2))
    h = (np.sin((dec_2 - dec_1) / 2) ** 2
         + np.cos(dec_1) * np.cos(dec_2) * np.sin((ra_2 - ra_1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def camera_centres(sector_camera_pairs):
    """{(sector, camera): (ra, dec)} from the TESS pointing table."""
    from tess_stars2px import TESS_Spacecraft_Pointing_Data
    sc = TESS_Spacecraft_Pointing_Data()
    out = {}
    for sector, camera in sorted(set(sector_camera_pairs)):
        idx = int(np.where(sc.sectors == int(sector))[0][0])
        out[(int(sector), int(camera))] = (float(sc.camRa[camera - 1, idx]),
                                           float(sc.camDec[camera - 1, idx]))
    return out


def assign_area(sector, camera, ccd, ra, dec):
    """camera*100 + ccd*10 + ring, ring 1..4 outward from the boresight."""
    sector = np.asarray(sector, dtype=np.int64)
    camera = np.asarray(camera, dtype=np.int64)
    ccd = np.asarray(ccd, dtype=np.int64)
    centres = camera_centres(zip(sector.tolist(), camera.tolist()))
    dist = np.full(len(ra), np.nan)
    for (sec, cam), (cra, cdec) in centres.items():
        sel = (sector == sec) & (camera == cam)
        dist[sel] = sphere_distance(np.asarray(ra)[sel], np.asarray(dec)[sel], cra, cdec)
    ring = np.full(len(ra), 4, dtype=np.int64)
    ring[dist < 0.75 * RADIUS_CAMERA] = 3
    ring[dist < 0.50 * RADIUS_CAMERA] = 2
    ring[dist < 0.25 * RADIUS_CAMERA] = 1
    area = camera * 100 + ccd * 10 + ring
    area[~np.isfinite(dist)] = -1
    return area.astype(np.int64), dist.astype(np.float32)


def make_tic_split(tics, val_frac, test_frac, seed):
    """Deterministic TIC-disjoint split. Sort unique TICs, then permute."""
    unique = np.sort(np.unique(np.asarray(tics, dtype=str)))
    order = np.random.default_rng(seed).permutation(len(unique))
    unique = unique[order]
    n = len(unique)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    assign = {}
    for t in unique[:n_test]:
        assign[t] = SPLIT_TEST
    for t in unique[n_test:n_test + n_val]:
        assign[t] = SPLIT_VAL
    for t in unique[n_test + n_val:]:
        assign[t] = SPLIT_TRAIN
    return np.array([assign[t] for t in np.asarray(tics, dtype=str)], dtype=np.uint8)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ build
def build_cache(cfg: DataConfig, verbose=True):
    """Parquet -> normalised gridded arrays + split manifest. Returns the audit."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(cfg.parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet matched {cfg.parquet_glob!r}")

    audit = {"files": len(files), "rows_read": 0, "dropped": {}}
    drop = {"malformed": 0, "no_surviving_cadence": 0, "too_few_observed": 0,
            "duplicate_tic": 0, "zero_mad": 0}

    tic, sector, camera, ccd, ra, dec = [], [], [], [], [], []
    times, fluxes = [], []
    seen = set()

    need = ["time", "flux", "TIC", "sector", "camera", "ccd", "ra", "dec",
            "TESS_flags", "TGLC_flags"]
    for path in files:
        tbl = pq.read_table(path, columns=need).to_pydict()
        n = len(tbl["TIC"])
        audit["rows_read"] += n
        for i in range(n):
            t = np.asarray(tbl["time"][i], dtype=np.float64)
            f = np.asarray(tbl["flux"][i], dtype=np.float64)
            tf = tbl["TESS_flags"][i]
            gf = tbl["TGLC_flags"][i]
            if t.size == 0 or f.size != t.size or tf is None or gf is None \
               or len(tf) != t.size or len(gf) != t.size:
                drop["malformed"] += 1
                continue
            keep = quality_keep(tf, gf, f, cfg.strict_quality)
            if keep.sum() < 2:
                drop["no_surviving_cadence"] += 1
                continue
            key = str(tbl["TIC"][i])
            if key in seen:
                drop["duplicate_tic"] += 1
                continue
            seen.add(key)
            fz, _, mad = normalize_median_mad(f[keep])
            if mad <= 0:
                drop["zero_mad"] += 1
            times.append(t[keep])
            fluxes.append(fz)
            tic.append(key)
            sector.append(int(tbl["sector"][i]))
            camera.append(int(tbl["camera"][i]))
            ccd.append(int(tbl["ccd"][i]))
            ra.append(float(tbl["ra"][i]))
            dec.append(float(tbl["dec"][i]))

    if not tic:
        raise ValueError("every row was dropped -- check the quality gates")

    sector = np.asarray(sector, dtype=np.int64)

    # Grid each sector on its own global time range, then stack.
    X = np.zeros((len(tic), cfg.seq_len), dtype=np.float32)
    M = np.zeros_like(X)
    sector_bounds = {}
    for sec in np.unique(sector):
        sel = np.flatnonzero(sector == sec)
        xs, ms, bounds = build_shared_sector([times[i] for i in sel],
                                             [fluxes[i] for i in sel], cfg.seq_len)
        X[sel], M[sel] = xs, ms
        sector_bounds[int(sec)] = bounds

    # Drop curves that kept too few bins.
    frac = M.mean(axis=1)
    good = frac >= cfg.min_observed_frac
    drop["too_few_observed"] = int((~good).sum())

    tic = np.asarray(tic, dtype=object)[good]
    X, M = X[good], M[good]
    sector = sector[good]
    camera = np.asarray(camera, dtype=np.int64)[good]
    ccd = np.asarray(ccd, dtype=np.int64)[good]
    ra = np.asarray(ra, dtype=np.float64)[good]
    dec = np.asarray(dec, dtype=np.float64)[good]

    area, cam_dist = assign_area(sector, camera, ccd, ra, dec)

    # Optional physics labels (downstream evaluation only).
    label = np.full(len(tic), -1, dtype=np.int16)
    label_map = {}
    if cfg.label_csv:
        import pandas as pd
        lab = pd.read_csv(cfg.label_csv, dtype={"TIC": str})
        col = "label" if "label" in lab.columns else lab.columns[1]
        classes = sorted(lab[col].astype(str).unique())
        label_map = {c: i for i, c in enumerate(classes)}
        by_tic = dict(zip(lab["TIC"].astype(str), lab[col].astype(str)))
        for i, t in enumerate(tic):
            if t in by_tic:
                label[i] = label_map[by_tic[t]]
    if cfg.require_labels and (label < 0).all():
        raise ValueError("require_labels set but no TIC matched the label table")

    split = make_tic_split(tic, cfg.val_frac, cfg.test_frac, cfg.split_seed)

    # Region eligibility, per (area, split): Part 1 needs group_size distinct stars.
    eligible = {}
    for a in np.unique(area):
        if a < 0:
            continue
        for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            k = int(((area == a) & (split == s)).sum())
            eligible[f"{int(a)}/{SPLIT_NAMES[s]}"] = k

    usable = {s: sorted(int(a) for a in np.unique(area)
                        if a >= 0 and eligible.get(f"{int(a)}/{SPLIT_NAMES[s]}", 0) >= cfg.group_size)
              for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    np.savez_compressed(
        cfg.out, flux=X, observed=M.astype(np.uint8), tic=tic.astype(str),
        sector=sector.astype(np.int16), camera=camera.astype(np.int16),
        ccd=ccd.astype(np.int16), area=area.astype(np.int32),
        ra=ra.astype(np.float32), dec=dec.astype(np.float32),
        cam_dist=cam_dist, label=label, split=split)

    audit.update({
        "curves_kept": int(len(tic)),
        "unique_tics": int(len(set(tic.tolist()))),
        "dropped": drop,
        "observed_frac": {"min": float(frac[good].min()), "median": float(np.median(frac[good])),
                          "max": float(frac[good].max())},
        "split_counts": {SPLIT_NAMES[s]: int((split == s).sum()) for s in (0, 1, 2)},
        "areas_total": int((np.unique(area) >= 0).sum()),
        "areas_usable_for_groups": {SPLIT_NAMES[s]: usable[s] for s in (0, 1, 2)},
        "sector_time_bounds": sector_bounds,
        "labels_present": int((label >= 0).sum()),
        "label_classes": label_map,
        "config": asdict(cfg),
        "git_sha": git_sha(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "npz_sha256": _sha256(cfg.out),
    })

    manifest = os.path.splitext(cfg.out)[0] + "_manifest.json"
    tic_lists = {SPLIT_NAMES[s]: sorted(tic[split == s].tolist()) for s in (0, 1, 2)}
    with open(manifest, "w") as fh:
        json.dump({**audit, "tics": tic_lists}, fh, indent=2)
    audit["manifest"] = manifest

    if verbose:
        print(json.dumps({k: v for k, v in audit.items() if k != "sector_time_bounds"},
                         indent=2, default=str))
    return audit


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ source
class RealCurveSource:
    """Loads a built cache and exposes one split. Mirrors SyntheticCurveSource."""

    def __init__(self, npz_path, split="train", require_label=False):
        d = np.load(npz_path, allow_pickle=False)
        code = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST,
                "all": None}[split]
        sel = slice(None) if code is None else np.flatnonzero(d["split"] == code)
        if require_label:
            base = np.arange(len(d["split"])) if code is None else sel
            sel = base[d["label"][base] >= 0]
        self.split_name = split
        self.flux = d["flux"][sel]
        self.observed = d["observed"][sel]
        self.tic = d["tic"][sel]
        self.sector = d["sector"][sel]
        self.camera = d["camera"][sel]
        self.ccd = d["ccd"][sel]
        self.area = d["area"][sel]
        self.ra = d["ra"][sel]
        self.dec = d["dec"][sel]
        self.label = d["label"][sel]
        self.region = self.area                 # RegionGroupDataset reads `region`
        if len(self.flux) == 0:
            raise ValueError(f"split {split!r} is empty in {npz_path}")

    def __len__(self):
        return len(self.flux)

    def describe(self):
        return {"split": self.split_name, "curves": len(self.flux),
                "unique_tics": int(len(set(self.tic.tolist()))),
                "areas": int(len(np.unique(self.area))),
                "labelled": int((self.label >= 0).sum())}


def assert_tic_disjoint(*sources):
    """No star may appear in more than one split."""
    sets = [set(s.tic.tolist()) for s in sources]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"{sources[i].split_name}/{sources[j].split_name} share "
                    f"{len(overlap)} TICs, e.g. {sorted(overlap)[:5]}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the real-curve cache.")
    for f in DataConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}",
                        type=type(f.default), default=f.default)
    args = ap.parse_args()
    build_cache(DataConfig(**{k: v for k, v in vars(args).items()}))
