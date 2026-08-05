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

    # one chip (default)
    python -m disentangle_attempt.fetch_data
    # the whole primary mission, four chips per sector
    SECTORS=$(seq -s, 1 26) CHIPS=1-1,2-2,3-3,4-4 N_STARS=1500 \
        python -m disentangle_attempt.fetch_data

Env: DA_DATA_DIR, SECTORS, CHIPS, N_STARS (per sector per chip), N_WORKERS, MIN_POINTS.
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

# SECTORS/CHIPS drive the multi-chip fetch. Sectors 1-26 are the primary mission and
# are ALL 30-minute FFIs, so their cadence grids are directly comparable; sector 27+
# switch to 10-minute and must not be mixed in.
SECTORS = tuple(int(v) for v in os.environ.get("SECTORS", "1").split(",") if v.strip())
CHIPS = tuple(tuple(int(x) for x in pair.split("-"))
              for pair in os.environ.get("CHIPS", "4-2").split(",") if pair.strip())
SECTOR_A = int(os.environ.get("SECTOR_A", str(SECTORS[0])))
CAMERA = int(os.environ.get("CAMERA", str(CHIPS[0][0])))   # CVZ camera in sectors 1-13
CCD = int(os.environ.get("CCD", str(CHIPS[0][1])))
N_STARS = int(os.environ.get("N_STARS", "2000"))           # per sector per chip
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
def fetch_chip(sector, camera, ccd, n_stars):
    """Download and extract one sector/camera/CCD block; returns a DataFrame."""
    patch = collect_patch(sector, camera, ccd, n_stars)
    if not patch:
        print(f"  s{sector:04d} cam{camera}-ccd{ccd}: no index lines, skipping", flush=True)
        return None
    print(f"s{sector:04d} cam{camera}-ccd{ccd}: {len(patch)} index lines", flush=True)
    paths = download_many([rel for _, rel in patch], f"s{sector:04d} cam{camera}-ccd{ccd}")
    if not paths:
        return None
    frame = extract_many(paths)
    if frame.empty:
        return None
    print(f"  -> {len(frame)} curves extracted", flush=True)
    return frame


def main():
    os.makedirs(FITS_DIR, exist_ok=True)
    if os.path.exists(OUT_PARQUET):
        print(f"{OUT_PARQUET} already exists -- nothing to do")
        return

    if any(s > 26 for s in SECTORS):
        raise SystemExit("sectors 27+ are 10-minute FFIs; do not mix them with 1-26")

    frames = []
    for sector in SECTORS:
        for camera, ccd in CHIPS:
            frame = fetch_chip(sector, camera, ccd, N_STARS)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise SystemExit("no usable curves downloaded")
    frame_a = pd.concat(frames, ignore_index=True)
    print(f"total: {len(frame_a)} curves over {len(SECTORS)} sectors x {len(CHIPS)} chips",
          flush=True)

    # tess-point per sector: DETECTOR_X/Y must come from the star's OWN sector solution.
    positions = []
    for sector, group in frame_a.groupby("sector"):
        table = tess_point_all(group["GAIADR3"].to_numpy(), group["ra"].to_numpy(),
                               group["dec"].to_numpy())
        positions.append(table[table["sector"] == int(sector)])
    positions = pd.concat(positions, ignore_index=True).drop_duplicates(
        ["GAIADR3", "sector", "camera", "ccd"])

    frame_a["TIC"] = frame_a["TIC"].astype(str)
    merged = frame_a.merge(positions, on=["GAIADR3", "sector", "camera", "ccd"], how="left")
    assert len(merged) == len(frame_a), "row count changed on detector-position merge"
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
    counts = merged.groupby(["sector", "camera", "ccd"]).size()
    print(f"wrote {OUT_PARQUET}: {len(merged)} rows, {merged['TIC'].nunique()} TICs, "
          f"{len(counts)} chips", flush=True)
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    main()
