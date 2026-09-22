"""Read TGLC FITS light curves straight into the arrays the model expects.

Same contract as real_data.build_cache (raw aperture_flux, the TESS/TGLC quality
gates, median/MAD normalisation, one shared per-sector 1024-bin grid), for
ad-hoc targets downloaded outside the parquet pipeline.
"""
import glob
import numpy as np

from .real_data import (quality_keep, normalize_median_mad, build_shared_sector,
                        GRID_1024)


def read_one(path):
    from astropy.io import fits
    with fits.open(path) as h:
        hdr, d = h[0].header, h[1].data
        return {
            "tic": str(hdr.get("TICID", "")),
            "gaia": str(hdr.get("GAIADR3", "")),
            "sector": int(hdr.get("SECTOR", -1)),
            "camera": int(hdr.get("CAMERA", -1)),
            "ccd": int(hdr.get("CCD", -1)),
            "ra": float(hdr.get("RA_OBJ", np.nan)),
            "dec": float(hdr.get("DEC_OBJ", np.nan)),
            "tmag": float(hdr.get("TESSMAG", np.nan)),
            "time": np.asarray(d["time"], dtype=np.float64),
            "flux": np.asarray(d["aperture_flux"], dtype=np.float64),
            "tess_flags": np.asarray(d["TESS_flags"], dtype=np.int64),
            "tglc_flags": np.asarray(d["TGLC_flags"], dtype=np.int64),
        }


def build_group(paths, strict=False, seq_len=GRID_1024, min_observed_frac=0.25):
    """FITS paths -> (flux, observed, meta). One shared grid across the group."""
    recs, times, fluxes = [], [], []
    for p in sorted(paths):
        try:
            r = read_one(p)
        except Exception:
            continue
        keep = quality_keep(r["tess_flags"], r["tglc_flags"], r["flux"], strict)
        if keep.sum() < 50:
            continue
        fz, _, _ = normalize_median_mad(r["flux"][keep])
        times.append(r["time"][keep])
        fluxes.append(fz)
        recs.append(r)
    if not recs:
        raise ValueError("no usable curve among the given files")

    X, M, bounds = build_shared_sector(times, fluxes, seq_len)
    good = M.mean(axis=1) >= min_observed_frac
    meta = {k: np.array([r[k] for r, g in zip(recs, good) if g])
            for k in ("tic", "gaia", "sector", "camera", "ccd", "ra", "dec", "tmag")}
    meta["time_bounds"] = bounds
    return X[good].astype(np.float32), M[good].astype(np.float32), meta


def order_by_separation(meta, ra0, dec0):
    """Indices sorted by angular distance from (ra0, dec0)."""
    from .real_data import sphere_distance
    d = sphere_distance(meta["ra"], meta["dec"], ra0, dec0)
    return np.argsort(d), d


def find_target(meta, tic):
    hit = np.flatnonzero(meta["tic"] == str(tic))
    if not len(hit):
        raise ValueError(f"TIC {tic} not among the loaded curves")
    return int(hit[0])


def build_cache_from_fits(dirs, out_npz, strict=True, seq_len=GRID_1024,
                          min_observed_frac=0.25, val_frac=0.15, test_frac=0.20,
                          split_seed=43, force_val_tics=(), label_csv=None):
    """FITS directories -> the same .npz layout real_data.build_cache writes.

    `force_val_tics` pins specific stars into the validation split so an
    evaluation target is never trained on.
    """
    import json, os
    from .real_data import (assign_area, make_tic_split, git_sha,
                            SPLIT_VAL, SPLIT_NAMES)

    paths = []
    for d in dirs:
        paths += glob.glob(os.path.join(d, "*.fits"))
    seen, recs, times, fluxes = set(), [], [], []
    dropped = {"unreadable": 0, "few_cadences": 0, "duplicate": 0}
    for p in sorted(paths):
        try:
            r = read_one(p)
        except Exception:
            dropped["unreadable"] += 1
            continue
        if r["gaia"] in seen:
            dropped["duplicate"] += 1
            continue
        keep = quality_keep(r["tess_flags"], r["tglc_flags"], r["flux"], strict)
        if keep.sum() < 50:
            dropped["few_cadences"] += 1
            continue
        seen.add(r["gaia"])
        fz, _, _ = normalize_median_mad(r["flux"][keep])
        times.append(r["time"][keep]); fluxes.append(fz); recs.append(r)

    X, M, bounds = build_shared_sector(times, fluxes, seq_len)
    good = M.mean(axis=1) >= min_observed_frac
    dropped["low_coverage"] = int((~good).sum())
    X, M = X[good], M[good]
    recs = [r for r, g in zip(recs, good) if g]

    col = lambda k, dt: np.array([r[k] for r in recs], dtype=dt)
    tic = col("tic", object).astype(str)
    sector, camera, ccd = col("sector", np.int64), col("camera", np.int64), col("ccd", np.int64)
    ra, dec = col("ra", np.float64), col("dec", np.float64)
    area, cam_dist = assign_area(sector, camera, ccd, ra, dec)

    # PhyTS labels + their own split, matched by GaiaDR3 (TIC is not unique --
    # TESS blends map one TIC to several Gaia sources).
    label = np.full(len(tic), -1, np.int16)
    classes, split = {}, None
    if label_csv:
        import pandas as pd
        lab = pd.read_csv(label_csv)
        gaia = col("gaia", object).astype(str)
        lab["GaiaID"] = lab["GaiaID"].astype(str)
        lab = lab.drop_duplicates("GaiaID")
        classes = {c: i for i, c in enumerate(sorted(lab["label"].astype(str).unique()))}
        by_gaia = dict(zip(lab["GaiaID"], lab["label"].astype(str)))
        for i, g in enumerate(gaia):
            if g in by_gaia:
                label[i] = classes[by_gaia[g]]
        if "split" in lab.columns:
            code = {"train": 0, "val": SPLIT_VAL, "test": 2}
            sp_by_gaia = dict(zip(lab["GaiaID"], lab["split"].astype(str)))
            split = make_tic_split(tic, val_frac, test_frac, split_seed)
            for i, g in enumerate(gaia):
                if g in sp_by_gaia and sp_by_gaia[g] in code:
                    split[i] = code[sp_by_gaia[g]]
    if split is None:
        split = make_tic_split(tic, val_frac, test_frac, split_seed)

    # TESS blends map one TIC to several Gaia sources, so a Gaia-keyed split can
    # put two sources at the same sky position in different splits. That is a
    # leak. Assign every conflicting TIC wholly to its most restrictive split
    # (test > val > train) so the evaluation sets keep their stars and the
    # duplicate leaves training.
    conflicts = 0
    order = {2: 0, SPLIT_VAL: 1, 0: 2}          # lower rank wins
    import collections as _c
    by_tic = _c.defaultdict(list)
    for i, t in enumerate(tic):
        by_tic[t].append(i)
    for t, idxs in by_tic.items():
        if len(idxs) > 1 and len({int(split[i]) for i in idxs}) > 1:
            win = min((int(split[i]) for i in idxs), key=lambda s: order[s])
            for i in idxs:
                split[i] = win
            conflicts += 1

    for t in force_val_tics:
        split[tic == str(t)] = SPLIT_VAL

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, flux=X, observed=M.astype(np.uint8), tic=tic,
                        sector=sector.astype(np.int16), camera=camera.astype(np.int16),
                        ccd=ccd.astype(np.int16), area=area.astype(np.int32),
                        ra=ra.astype(np.float32), dec=dec.astype(np.float32),
                        cam_dist=cam_dist, label=label, split=split)
    audit = {"files": len(paths), "curves_kept": int(len(tic)), "dropped": dropped,
             "strict_quality": strict,
             "areas": {int(a): int((area == a).sum()) for a in np.unique(area)},
             "split_counts": {SPLIT_NAMES[s]: int((split == s).sum()) for s in (0, 1, 2)},
             "forced_val": list(force_val_tics), "time_bounds": bounds,
             "label_classes": classes,
             "tic_split_conflicts_resolved": conflicts,
             "labelled": int((label >= 0).sum()),
             "label_counts": {str(k): int((label == v).sum()) for k, v in classes.items()},
             "git_sha": git_sha()}
    with open(os.path.splitext(out_npz)[0] + "_manifest.json", "w") as fh:
        json.dump(audit, fh, indent=2)
    return audit
