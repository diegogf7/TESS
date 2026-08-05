"""Acquire a cross-sector TGLC patch: the same stars in two sectors, one camera/CCD.

Why this exists: the repository's existing parquets are single-sector (Sector 14
dense_v2) or a sky-uniform reservoir sample across sectors 1-26, in which the same
TIC essentially never appears twice. The cross-sector branch of this experiment
needs the SAME star in two sectors, so a purpose-built patch is downloaded here.

How a compact, cross-sector-rich patch is found without scanning multi-GB indexes:

1. MAST's per-sector TGLC bulk-download script is sorted by camera/CCD and then by
   Gaia source id. It supports HTTP range requests, so a binary search over byte
   offsets lands on the requested camera/CCD block after ~25 4 KB reads.
2. Gaia DR3 source ids are HEALPix-ordered, so a run of consecutive lines is a
   spatially compact patch -- exactly the dense detector neighbourhood the peer
   selection wants.
3. Camera 4 in sectors 1-13 points at the southern continuous viewing zone, so the
   same stars are re-observed in most of those sectors. tess-point turns each star's
   RA/Dec into (sector, camera, CCD, detector column/row) for EVERY sector at once,
   which both picks the partner sector and supplies DETECTOR_X/DETECTOR_Y.
4. TGLC file paths are fully determined by the Gaia id: the four 4-digit directories
   are ``f"{gaia_id // 100000:016d}"`` split into groups of four. So the partner
   sector's URLs are constructed directly, with no second index scan.

    python -m disentangle_attempt.fetch_data
Env: DA_DATA_DIR, SECTOR_A, CAMERA, CCD, N_STARS, N_WORKERS, MIN_POINTS.
"""

import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DA_DATA_DIR", os.path.join(HERE, "data"))
FITS_DIR = os.path.join(DATA_DIR, "fits")
OUT_PARQUET = os.path.join(DATA_DIR, "cross_sector_raw.parquet")

SECTOR_A = int(os.environ.get("SECTOR_A", "1"))
CAMERA = int(os.environ.get("CAMERA", "4"))          # CVZ camera in sectors 1-13
CCD = int(os.environ.get("CCD", "2"))
N_STARS = int(os.environ.get("N_STARS", "2000"))
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))
MIN_POINTS = int(os.environ.get("MIN_POINTS", "200"))
CANDIDATE_SECTORS = tuple(range(1, 27))              # TGLC v1 primary mission
# The model uses ONE curve per star, so the partner-sector download is optional and
# off by default. Set PARTNER_SECTOR=1 to also fetch the same TICs in a second
# sector (needed only for cross-sector experiments).
WANT_PARTNER = os.environ.get("PARTNER_SECTOR", "0") == "1"

BULK = ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
        "hlsp_tglc_tess_ffi_s{sector:04d}_tess_v1_llc.sh")
MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tglc/{rel}"
LINE_PATH = re.compile(r"--output '([^']+)'")
CAMCCD = re.compile(r"cam(\d)-ccd(\d)")
GAIAID = re.compile(r"gaiaid-(\d+)-")

# Mirrors src/tglc/extract_raw_parquet_cadence.py: the cadence-aligned pipeline needs
# these four beyond time/flux. background feeds the quiet-reference ranking.
EXTRA_COLUMNS = ("cadence_num", "TESS_flags", "TGLC_flags", "background")


# ------------------------------------------------------------------ index access
def http_get(url, byte_range=None, timeout=60, retries=3):
    request = urllib.request.Request(url)
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:                     # transient MAST 5xx / timeouts
            last = exc
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def content_length(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def line_at(url, offset, size=4096):
    """First COMPLETE line at or after `offset` (the partial head line is dropped)."""
    chunk = http_get(url, (offset, offset + size)).decode("utf-8", "replace")
    parts = chunk.split("\n")
    if offset > 0:
        parts = parts[1:]
    for part in parts:
        if "--output" in part:
            return part
    return None


def camccd_key(line):
    hit = CAMCCD.search(line or "")
    return (int(hit.group(1)), int(hit.group(2))) if hit else None


def find_block_offset(url, size, target):
    """Byte offset of the first line whose (camera, ccd) >= target (sorted file)."""
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) // 2
        key = camccd_key(line_at(url, mid))
        if key is None or key < target:             # header line sorts first
            lo = mid + 1
        else:
            hi = mid
    return max(lo - 4096, 0)


def collect_patch(sector, camera, ccd, n_stars):
    """Consecutive bulk-script lines for one camera/CCD -> [(gaia_id, rel_path)]."""
    url = BULK.format(sector=sector)
    size = content_length(url)
    start = find_block_offset(url, size, (camera, ccd))
    rows, offset = [], start
    # ~370 bytes/line, so 2 MB per read is ~5.5k candidate lines
    while len(rows) < n_stars and offset < size:
        text = http_get(url, (offset, min(offset + 2_000_000, size))).decode("utf-8", "replace")
        lines = text.split("\n")
        offset += 2_000_000
        for line in lines[1:-1] if offset > start else lines[:-1]:
            key = camccd_key(line)
            if key is None:
                continue
            if key < (camera, ccd):
                continue
            if key > (camera, ccd):                 # walked past the block
                return rows
            path = LINE_PATH.search(line)
            gaia = GAIAID.search(line)
            if path and gaia:
                rows.append((int(gaia.group(1)), path.group(1)))
                if len(rows) >= n_stars:
                    return rows
    return rows


def tglc_rel_path(gaia_id, sector, camera, ccd):
    """TGLC relative path is a pure function of (gaia id, sector, camera, ccd)."""
    digits = f"{int(gaia_id) // 100000:016d}"
    dirs = "/".join(digits[i:i + 4] for i in range(0, 16, 4))
    name = (f"hlsp_tglc_tess_ffi_gaiaid-{gaia_id}-s{sector:04d}"
            f"-cam{camera}-ccd{ccd}_tess_v1_llc.fits")
    return f"s{sector:04d}/cam{camera}-ccd{ccd}/{dirs}/{name}"


# -------------------------------------------------------------------- downloading
def download_one(rel):
    local = os.path.join(FITS_DIR, rel)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        payload = http_get(MAST_FILE.format(rel=rel), retries=2)
    except Exception:
        return None                                  # star absent from that sector
    if len(payload) < 10_000:                        # HTML error page, not a FITS
        return None
    tmp = local + ".part"
    with open(tmp, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, local)
    return local


def download_many(rels, label):
    print(f"downloading {len(rels)} {label} light curves with {N_WORKERS} workers", flush=True)
    with ThreadPoolExecutor(N_WORKERS) as pool:
        paths = list(pool.map(download_one, rels))
    ok = [p for p in paths if p]
    print(f"  {len(ok)}/{len(rels)} retrieved", flush=True)
    return ok


# -------------------------------------------------------------------- extraction
def native(array):
    """FITS columns are big-endian; parquet only accepts native byte order."""
    array = np.asarray(array)
    return array.astype(array.dtype.newbyteorder("="), copy=False)


def extract_one(path):
    """One FITS -> one record. Column contract mirrors extract_raw_parquet_cadence.py."""
    try:
        with fits.open(path, memmap=False) as hdul:
            data, head = hdul[1].data, hdul[0].header
            available = set(data.columns.names)
            time = np.asarray(data["time"], dtype=np.float64)
            raw = np.asarray(data["aperture_flux"], dtype=np.float64)
            good = np.isfinite(time) & np.isfinite(raw)
            if good.sum() < MIN_POINTS:
                return None
            row = {
                "time": time[good], "flux": raw[good],
                "TIC": str(head.get("TICID", "")),
                "GAIADR3": int(head.get("GAIADR3", -1)),
                "sector": int(head.get("SECTOR", -1)),
                "camera": int(head.get("CAMERA", -1)),
                "ccd": int(head.get("CCD", -1)),
                "ra": float(head.get("RA_OBJ", float("nan"))),
                "dec": float(head.get("DEC_OBJ", float("nan"))),
            }
            for col in EXTRA_COLUMNS:
                row[col] = native(data[col])[good] if col in available else None
            return row
    except Exception:
        return None


def extract_many(paths):
    with ThreadPoolExecutor(N_WORKERS) as pool:
        rows = [r for r in pool.map(extract_one, paths) if r is not None]
    return pd.DataFrame(rows)


# ------------------------------------------------------------ detector positions
def tess_point_all(gaia_ids, ra, dec):
    """(gaia_id, sector, camera, ccd, DETECTOR_X, DETECTOR_Y) for every observation.

    Same source of truth as src/tglc/merge_detector_positions.py: official TESS CCD
    column/row from tess-point, never the aperture-local STAR_X/STAR_Y.
    """
    from tess_stars2px import tess_stars2px_function_entry
    index = np.arange(len(gaia_ids))
    oid, _, _, sec, cam, ccd, col, row, _ = tess_stars2px_function_entry(
        index, np.asarray(ra, float), np.asarray(dec, float))
    table = pd.DataFrame({
        "oid": np.asarray(oid, int), "sector": np.asarray(sec, int),
        "camera": np.asarray(cam, int), "ccd": np.asarray(ccd, int),
        "DETECTOR_X": np.asarray(col, float), "DETECTOR_Y": np.asarray(row, float)})
    table["GAIADR3"] = np.asarray(gaia_ids, np.int64)[table["oid"].to_numpy()]
    return table.drop(columns=["oid"]).drop_duplicates(["GAIADR3", "sector", "camera", "ccd"])


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(FITS_DIR, exist_ok=True)
    if os.path.exists(OUT_PARQUET):
        print(f"{OUT_PARQUET} already exists -- nothing to do")
        return

    patch = collect_patch(SECTOR_A, CAMERA, CCD, N_STARS)
    print(f"sector {SECTOR_A} cam{CAMERA}-ccd{CCD}: {len(patch)} consecutive index lines", flush=True)
    paths_a = download_many([rel for _, rel in patch], f"sector-{SECTOR_A}")
    frame_a = extract_many(paths_a)
    if frame_a.empty:
        raise SystemExit("no usable sector-A curves")
    print(f"sector {SECTOR_A}: {len(frame_a)} curves extracted", flush=True)

    positions = tess_point_all(frame_a["GAIADR3"].to_numpy(),
                               frame_a["ra"].to_numpy(), frame_a["dec"].to_numpy())

    if not WANT_PARTNER:
        frame = frame_a.copy()
        frame["TIC"] = frame["TIC"].astype(str)
        merged = frame.merge(positions, on=["GAIADR3", "sector", "camera", "ccd"],
                             how="left")
        missing = int(merged["DETECTOR_X"].isna().sum())
        if missing:
            print(f"dropping {missing} rows without a tess-point solution", flush=True)
            merged = merged[merged["DETECTOR_X"].notna()].reset_index(drop=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        merged.to_parquet(OUT_PARQUET)
        print(f"wrote {OUT_PARQUET}: {len(merged)} rows, {merged['TIC'].nunique()} TICs "
              f"(single sector {SECTOR_A}; set PARTNER_SECTOR=1 for cross-sector data)",
              flush=True)
        return

    # Partner sector: the one re-observing the most of these stars (any camera/CCD).
    counts = (positions[(positions["sector"] != SECTOR_A)
                        & positions["sector"].isin(CANDIDATE_SECTORS)]
              .groupby("sector")["GAIADR3"].nunique().sort_values(ascending=False))
    print("cross-sector availability (top 5):\n" + counts.head().to_string(), flush=True)
    sector_b = int(counts.index[0])

    partner = positions[positions["sector"] == sector_b]
    partner = partner[partner["GAIADR3"].isin(set(frame_a["GAIADR3"].tolist()))]
    rels_b = [tglc_rel_path(g, sector_b, c, d) for g, c, d
              in zip(partner["GAIADR3"], partner["camera"], partner["ccd"])]
    paths_b = download_many(rels_b, f"sector-{sector_b}")
    frame_b = extract_many(paths_b)
    print(f"sector {sector_b}: {len(frame_b)} curves extracted", flush=True)

    frame = pd.concat([frame_a, frame_b], ignore_index=True)
    frame["TIC"] = frame["TIC"].astype(str)
    merged = frame.merge(positions, on=["GAIADR3", "sector", "camera", "ccd"], how="left")
    assert len(merged) == len(frame), "row count changed on detector-position merge"
    missing = int(merged["DETECTOR_X"].isna().sum())
    if missing:
        print(f"dropping {missing} rows without a tess-point solution", flush=True)
        merged = merged[merged["DETECTOR_X"].notna()].reset_index(drop=True)

    for col in EXTRA_COLUMNS:
        n_missing = int(merged[col].isna().sum())
        if n_missing:
            print(f"WARNING: column {col} missing for {n_missing}/{len(merged)} rows", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_parquet(OUT_PARQUET)
    both = (merged.groupby("TIC")["sector"].nunique() >= 2).sum()
    print(f"wrote {OUT_PARQUET}: {len(merged)} rows, {merged['TIC'].nunique()} TICs, "
          f"{both} observed in both sectors (target {SECTOR_A}, partner {sector_b})", flush=True)


if __name__ == "__main__":
    main()
