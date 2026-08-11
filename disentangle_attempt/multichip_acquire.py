"""Acquire a full-chip TGLC data set across many sector/camera/CCD chips.

Why this is not :mod:`disentangle_attempt.fetch_data`.  ``fetch_data`` takes a run of
*consecutive* bulk-script lines.  Gaia source ids are HEALPix-ordered, so consecutive
lines are a spatially compact patch: the existing Sector-1/camera-4/CCD-2 parquet spans
only ``X [1724, 2091]``, ``Y [3, 222]`` -- a 367x219 pixel corner of a 2048x2048 CCD.
The two-band spread-peer rule needs peers at ``[128, 384)`` **and** ``[384, 768]``
pixels with a 256 px minimum peer-to-peer separation, which that corner cannot host at
all.  So this module indexes the *complete* per-chip block and samples it uniformly,
giving whole-chip coverage like ``acquire_spread_peer_data`` did for one chip.

The flux contract is unchanged from the trained two-decoder run: raw TGLC
``aperture_flux`` with ``cadence_num``/``TESS_flags``/``TGLC_flags``/``background``,
extracted by :func:`disentangle_attempt.fetch_data.extract_one`, and detector positions
from ``tess_stars2px`` via :func:`disentangle_attempt.fetch_data.tess_point_all`.  This
module downloads and arranges data; it never filters on curve values.

Five resumable stages, each safe to re-run::

    python -m disentangle_attempt.multichip_acquire --stage index
    python -m disentangle_attempt.multichip_acquire --stage select
    python -m disentangle_attempt.multichip_acquire --stage download
    python -m disentangle_attempt.multichip_acquire --stage extract
    python -m disentangle_attempt.multichip_acquire --stage positions
    python -m disentangle_attempt.multichip_acquire --stage all      # the default

Selection is deterministic and uses no curve value: a chip's complete Gaia id list is
ranked by ``sha256(f"{seed}:{sector}:{camera}:{ccd}:{gaia}")`` and the first
``--stars-per-chip`` ids are taken.  The complete block is the whole chip catalogue, so
a uniform sample of it covers the whole detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from disentangle_attempt.fetch_data import extract_one, tglc_rel_path


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data_multichip"
BULK_URL = (
    "https://archive.stsci.edu/hlsps/tglc/download_scripts/"
    "hlsp_tglc_tess_ffi_s{sector:04d}_tess_v1_llc.sh"
)
# Two routes to the same file.  The static archive path is what the acquisition uses:
# measured at 249 files/s with zero errors at 32-128 concurrent requests, whereas the
# API gateway rate-limits hard (2525 of 3000 requests answered HTTP 429 at 64 workers).
# The gateway is kept only as a per-file fallback.
ARCHIVE_FILE = "https://archive.stsci.edu/hlsps/tglc/{rel}"
MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tglc/{rel}"

# Sectors 1-26 are the 30-minute primary mission.  Sector 27+ switch to 10-minute FFIs
# and would land on an incompatible cadence grid, so they are refused outright.
MAX_PRIMARY_MISSION_SECTOR = 26
CAMERAS = (1, 2, 3, 4)
CCDS = (1, 2, 3, 4)
SEED = 42
# A FITS smaller than this is a MAST HTML error page, not a light curve.
MIN_FITS_BYTES = 10_000

CHIP_RE = re.compile(rb"-s(\d{4})-cam(\d)-ccd(\d)_")
GAIA_RE = re.compile(rb"gaiaid-(\d+)-")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chip_name(sector: int, camera: int, ccd: int) -> str:
    return f"s{sector:04d}_cam{camera}_ccd{ccd}"


def all_chips(
    sectors: Iterable[int], chips: str | None = None
) -> list[tuple[int, int, int]]:
    """Every (sector, camera, ccd) requested.

    ``chips`` filters to specific detectors, written ``"camera-ccd"`` and separated by
    commas (``"4-2"``, ``"1-1,1-2"``).  ``None`` or ``"all"`` means all sixteen.
    """
    if chips is None or str(chips).strip().lower() == "all":
        pairs = [(camera, ccd) for camera in CAMERAS for ccd in CCDS]
    else:
        pairs = []
        for token in str(chips).split(","):
            token = token.strip()
            if not token:
                continue
            camera, _, ccd = token.partition("-")
            pair = (int(camera), int(ccd))
            if pair[0] not in CAMERAS or pair[1] not in CCDS:
                raise ValueError(f"chip {token!r} is not a valid camera-ccd pair")
            pairs.append(pair)
        if not pairs:
            raise ValueError("no valid chips requested")
    return [(int(s), camera, ccd) for s in sectors for camera, ccd in pairs]


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(part, path)


# --------------------------------------------------------------- stage 1: indexing
def _open_bulk_stream(sector: int, timeout: int):
    url = BULK_URL.format(sector=sector)
    request = urllib.request.Request(
        url, headers={"User-Agent": "TESS-multichip-acquisition/1"}
    )
    return urllib.request.urlopen(request, timeout=timeout), url


def build_sector_index(
    data_dir: Path, sector: int, timeout: int = 120, retries: int = 3
) -> dict[str, object]:
    """Stream one sector's bulk script once and split it into 16 per-chip id lists.

    Only the Gaia id is stored.  The TGLC relative path is a pure function of
    ``(gaia_id, sector, camera, ccd)`` (see :func:`fetch_data.tglc_rel_path`), so
    keeping the path would multiply the index size for no information.
    """
    index_dir = data_dir / "index" / f"s{sector:04d}"
    marker = index_dir / "COMPLETE.json"
    if marker.is_file():
        return json.loads(marker.read_text())

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        # A partial stream leaves partial files, so every attempt starts clean.
        for stale in index_dir.glob("*.part"):
            stale.unlink()
        index_dir.mkdir(parents=True, exist_ok=True)
        handles = {
            (camera, ccd): (index_dir / f"cam{camera}-ccd{ccd}.txt.part").open("w")
            for camera in CAMERAS
            for ccd in CCDS
        }
        seen: dict[tuple[int, int], set[int]] = {key: set() for key in handles}
        counts = {key: 0 for key in handles}
        wrong_sector = duplicates = unparsed = 0
        started = time.time()
        consumed = 0
        try:
            stream, url = _open_bulk_stream(sector, timeout)
            declared = stream.headers.get("Content-Length")
            declared = int(declared) if declared is not None else None
            with stream:
                for line_number, raw in enumerate(stream, 1):
                    consumed += len(raw)
                    chip_hit = CHIP_RE.search(raw)
                    if chip_hit is None:
                        unparsed += 1
                        continue
                    line_sector = int(chip_hit.group(1))
                    key = (int(chip_hit.group(2)), int(chip_hit.group(3)))
                    if line_sector != sector:
                        wrong_sector += 1
                        continue
                    if key not in handles:
                        unparsed += 1
                        continue
                    gaia_hit = GAIA_RE.search(raw)
                    if gaia_hit is None:
                        unparsed += 1
                        continue
                    gaia = int(gaia_hit.group(1))
                    if gaia in seen[key]:
                        duplicates += 1
                        continue
                    seen[key].add(gaia)
                    handles[key].write(f"{gaia}\n")
                    counts[key] += 1
                    if line_number % 1_000_000 == 0:
                        print(
                            f"  s{sector:04d}: {line_number:,} lines, "
                            f"{sum(counts.values()):,} ids "
                            f"({time.time() - started:.0f}s)",
                            flush=True,
                        )
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            # A mid-file connection close is the dangerous failure: it ends the
            # iteration cleanly, so without these checks a truncated stream is written
            # as a COMPLETE index.  The bulk script is sorted by camera then CCD, so
            # truncation shows up as trailing all-zero chips.
            if declared is not None and consumed != declared:
                raise RuntimeError(
                    f"sector {sector} stream truncated: read {consumed:,} of "
                    f"{declared:,} bytes"
                )
            empty = [
                f"cam{camera}-ccd{ccd}"
                for camera, ccd in handles
                if counts[(camera, ccd)] == 0
            ]
            if empty:
                raise RuntimeError(
                    f"sector {sector} produced no ids for {len(empty)} of 16 chips "
                    f"({', '.join(empty)}); the bulk script is camera/CCD-sorted, so "
                    "this means the stream ended early"
                )
            for (camera, ccd) in handles:
                part = index_dir / f"cam{camera}-ccd{ccd}.txt.part"
                os.replace(part, index_dir / f"cam{camera}-ccd{ccd}.txt")
            metadata = {
                "created_utc": utc_now(),
                "sector": sector,
                "source_url": url,
                "source_bytes_read": int(consumed),
                "source_bytes_declared": declared,
                "seconds": time.time() - started,
                "chip_counts": {
                    f"cam{camera}-ccd{ccd}": counts[(camera, ccd)]
                    for camera, ccd in handles
                },
                "total_ids": int(sum(counts.values())),
                "duplicate_gaia_lines": duplicates,
                "other_sector_lines": wrong_sector,
                "unparsed_lines": unparsed,
            }
            atomic_write_json(marker, metadata)
            print(
                f"sector {sector}: indexed {metadata['total_ids']:,} ids over 16 chips "
                f"in {metadata['seconds']:.0f}s",
                flush=True,
            )
            return metadata
        except Exception as exc:  # transient MAST 5xx / connection resets
            last_error = exc
            for handle in handles.values():
                if not handle.closed:
                    handle.close()
            print(
                f"  sector {sector} index attempt {attempt}/{retries} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    raise RuntimeError(f"could not index sector {sector}: {last_error}")


# -------------------------------------------------------------- stage 2: selection
def selection_rank(sector: int, camera: int, ccd: int, gaia: int, seed: int) -> bytes:
    key = f"{seed}:{sector}:{camera}:{ccd}:{gaia}".encode("ascii")
    return hashlib.sha256(key).digest()


def select_chip(
    data_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    stars_per_chip: int,
    seed: int = SEED,
) -> Path:
    """Deterministic uniform sample of a chip's complete Gaia id list.

    No curve value, magnitude or morphology enters this choice.  The source list is the
    whole chip block, so a uniform sample of it is spread over the whole detector.
    """
    index_path = data_dir / "index" / f"s{sector:04d}" / f"cam{camera}-ccd{ccd}.txt"
    if not index_path.is_file():
        raise FileNotFoundError(f"chip index missing, run --stage index first: {index_path}")
    out_path = data_dir / "selection" / f"{chip_name(sector, camera, ccd)}.csv"
    if out_path.is_file():
        return out_path

    ids = np.loadtxt(index_path, dtype=np.int64, ndmin=1)
    if ids.size == 0:
        raise RuntimeError(f"chip index is empty: {index_path}")
    ranked = sorted(
        (int(value) for value in ids),
        key=lambda gaia: selection_rank(sector, camera, ccd, gaia, seed),
    )
    chosen = ranked[: int(stars_per_chip)]
    frame = pd.DataFrame(
        {
            "selection_rank": np.arange(len(chosen), dtype=np.int64),
            "GAIADR3": np.asarray(chosen, dtype=np.int64),
            "sector": sector,
            "camera": camera,
            "ccd": ccd,
        }
    )
    frame["rel_path"] = [
        tglc_rel_path(gaia, sector, camera, ccd) for gaia in frame["GAIADR3"]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    frame.to_csv(part, index=False)
    os.replace(part, out_path)
    print(
        f"{chip_name(sector, camera, ccd)}: sampled {len(frame):,} of {ids.size:,} "
        f"catalogue sources",
        flush=True,
    )
    return out_path


# -------------------------------------------------------------- stage 3: downloads
@dataclass(frozen=True)
class DownloadResult:
    rel: str
    local: str
    status: str


_SESSION_LOCK = threading.Lock()
_SESSION: dict[str, object] = {}


def _session(pool_size: int):
    """One shared pooled HTTPS session for all worker threads.

    Each light curve is only ~58 KB, so a fresh TCP+TLS handshake per file costs more
    than the transfer.  Keep-alive connection reuse is the whole win at 240k files.

    The pool is shared rather than thread-local on purpose: a session *per thread*,
    each sized to the worker count, opens up to ``workers**2`` sockets and exhausts the
    process file-descriptor limit long before the archive objects -- which looks
    exactly like server-side throttling but is entirely self-inflicted.
    """
    with _SESSION_LOCK:
        session = _SESSION.get("session")
        if session is None:
            import requests
            from requests.adapters import HTTPAdapter

            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0
            )
            session.mount("https://", adapter)
            session.headers["User-Agent"] = "TESS-multichip-acquisition/1"
            _SESSION["session"] = session
        return session


def local_fits_path(fits_dir: Path, sector: int, camera: int, ccd: int, rel: str) -> Path:
    """Flat per-chip layout: ``fits/<chip>/<filename>``.

    TGLC's own path is ``s0001/cam4-ccd2/0046/2961/9538/4089/<file>``, where the four
    nested directories come from the Gaia id.  A uniform sample of a chip shares almost
    no prefixes, so mirroring that tree costs ~5 inodes per light curve -- 240k files
    would need ~1.2M inodes and blow a 1M file quota long before running out of space.
    One directory per chip costs ~1 inode per file instead.
    """
    return fits_dir / chip_name(sector, camera, ccd) / Path(rel).name


def download_one(
    rel: str, local: Path, timeout: int, retries: int, pool_size: int = 16
) -> DownloadResult:
    if local.is_file() and local.stat().st_size >= MIN_FITS_BYTES:
        return DownloadResult(rel, str(local), "cached")
    local.parent.mkdir(parents=True, exist_ok=True)
    session = _session(pool_size)
    urls = (ARCHIVE_FILE.format(rel=rel), MAST_FILE.format(rel=rel))
    for attempt in range(retries):
        url = urls[min(attempt, len(urls) - 1)] if attempt else urls[0]
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return DownloadResult(rel, "", "absent")
            if response.status_code == 429:
                # Only the API gateway does this, but honour it rather than burning
                # the retry budget in a tight loop.
                delay = float(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(delay, 30.0))
                continue
            response.raise_for_status()
            payload = response.content
        except Exception:
            continue
        if len(payload) < MIN_FITS_BYTES:
            # A missing file can come back as a short HTML page, not an HTTP error.
            return DownloadResult(rel, "", "absent")
        part = local.with_name(local.name + ".part")
        part.write_bytes(payload)
        os.replace(part, local)
        return DownloadResult(rel, str(local), "downloaded")
    return DownloadResult(rel, "", "failed")


def purge_chip_fits(data_dir: Path, sector: int, camera: int, ccd: int) -> int:
    """Delete one chip's FITS once its parquet chunk exists.  Returns files removed."""
    chunk = data_dir / "chunks" / f"{chip_name(sector, camera, ccd)}.parquet"
    if not chunk.is_file():
        raise RuntimeError(
            f"refusing to purge {chip_name(sector, camera, ccd)} FITS: "
            f"no parquet chunk at {chunk}"
        )
    directory = data_dir / "fits" / chip_name(sector, camera, ccd)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.rglob("*"):
        if path.is_file():
            path.unlink()
            removed += 1
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    directory.rmdir()
    return removed


def download_chip(
    data_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    workers: int,
    timeout: int = 60,
    retries: int = 3,
    purge_fits: bool = False,
) -> dict[str, object]:
    name = chip_name(sector, camera, ccd)
    marker = data_dir / "download" / f"{name}.json"
    if marker.is_file():
        return json.loads(marker.read_text())
    selection = pd.read_csv(data_dir / "selection" / f"{name}.csv")
    fits_dir = data_dir / "fits"
    started = time.time()
    tally = {"cached": 0, "downloaded": 0, "absent": 0, "failed": 0}
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(
                download_one,
                rel,
                local_fits_path(fits_dir, sector, camera, ccd, rel),
                timeout,
                retries,
                workers,
            )
            for rel in selection["rel_path"]
        ]
        for future in as_completed(futures):
            tally[future.result().status] += 1
    summary = {
        "created_utc": utc_now(),
        "chip": name,
        "requested": int(len(selection)),
        "seconds": time.time() - started,
        **tally,
    }
    retrieved = tally["cached"] + tally["downloaded"]
    if retrieved == 0:
        raise RuntimeError(f"{name}: no light curves retrieved")
    # "absent" is a settled answer from MAST, but "failed" means the network gave up.
    # Only a run with no such failures earns the marker, so re-running this stage
    # retries them instead of skipping the chip.  Files already on disk are detected
    # and reused, so a retry costs only the missing ones.
    if tally["failed"] == 0:
        atomic_write_json(marker, summary)
    note = "" if tally["failed"] == 0 else "  [incomplete: re-run to retry]"
    # Extract and delete this chip's FITS before moving on.  Holding all 80 chips of
    # raw FITS at once is what exhausts a file-count quota; one chip at a time is
    # ~3000 inodes, and re-downloading a chip costs only a few seconds.
    if purge_fits and tally["failed"] == 0:
        extract_chip(data_dir, sector, camera, ccd, workers)
        removed = purge_chip_fits(data_dir, sector, camera, ccd)
        summary["purged_fits"] = removed
        note += f"  [extracted, {removed:,} FITS purged]"
    print(
        f"{name}: {retrieved:,}/{len(selection):,} retrieved "
        f"({tally['absent']} absent, {tally['failed']} failed) "
        f"in {summary['seconds']:.0f}s" + note,
        flush=True,
    )
    return summary


# ------------------------------------------------------------- stage 4: extraction
def extract_chip(data_dir: Path, sector: int, camera: int, ccd: int, workers: int) -> Path | None:
    """One chip's FITS -> one parquet chunk, using the shared aperture_flux reader."""
    name = chip_name(sector, camera, ccd)
    chunk = data_dir / "chunks" / f"{name}.parquet"
    if chunk.is_file():
        return chunk
    selection = pd.read_csv(data_dir / "selection" / f"{name}.csv")
    fits_dir = data_dir / "fits"
    paths = [
        str(path)
        for path in (
            local_fits_path(fits_dir, sector, camera, ccd, rel)
            for rel in selection["rel_path"]
        )
        if path.is_file()
    ]
    if not paths:
        print(f"{name}: no FITS on disk, skipping", flush=True)
        return None
    with ThreadPoolExecutor(workers) as pool:
        rows = [row for row in pool.map(extract_one, paths) if row is not None]
    if not rows:
        print(f"{name}: nothing extracted, skipping", flush=True)
        return None
    frame = pd.DataFrame(rows)
    frame["TIC"] = frame["TIC"].astype(str)
    # The FITS header is authoritative for chip membership; tess-point disagreement is
    # handled separately in the position stage.
    frame = frame[
        (frame["sector"] == sector)
        & (frame["camera"] == camera)
        & (frame["ccd"] == ccd)
    ].reset_index(drop=True)
    if frame.empty:
        print(f"{name}: no rows matched the requested chip, skipping", flush=True)
        return None
    chunk.parent.mkdir(parents=True, exist_ok=True)
    part = chunk.with_name(chunk.name + ".part")
    frame.to_parquet(part)
    os.replace(part, chunk)
    print(f"{name}: extracted {len(frame):,} curves", flush=True)
    return chunk


# -------------------------------------------------- stage 5: detector positions
def resolve_sector_positions(data_dir: Path, sector: int, batch_size: int = 100_000) -> pd.DataFrame:
    """tess-point positions for one sector, computed from the extracted metadata only."""
    from disentangle_attempt.fetch_data import tess_point_all

    cache = data_dir / "positions" / f"s{sector:04d}.parquet"
    if cache.is_file():
        return pd.read_parquet(cache)
    chunks = sorted((data_dir / "chunks").glob(f"s{sector:04d}_cam*_ccd*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"no extracted chunks for sector {sector}")
    meta = pd.concat(
        [pd.read_parquet(path, columns=["GAIADR3", "ra", "dec"]) for path in chunks],
        ignore_index=True,
    ).drop_duplicates("GAIADR3")
    pieces = []
    for start in range(0, len(meta), batch_size):
        batch = meta.iloc[start : start + batch_size]
        table = tess_point_all(
            batch["GAIADR3"].to_numpy(np.int64),
            batch["ra"].to_numpy(float),
            batch["dec"].to_numpy(float),
            sector=sector,
        )
        pieces.append(table[table["sector"] == sector])
        print(
            f"  sector {sector}: {min(start + batch_size, len(meta)):,}/{len(meta):,} "
            f"positions resolved",
            flush=True,
        )
    positions = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["GAIADR3", "sector", "camera", "ccd"]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    positions.to_parquet(part)
    os.replace(part, cache)
    return positions


def merge_positions(data_dir: Path, sectors: Iterable[int]) -> dict[str, object]:
    """Attach DETECTOR_X/Y per chip and write the final parquet directory."""
    out_dir = data_dir / "cross_sector_raw.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for sector in sectors:
        positions = resolve_sector_positions(data_dir, sector)
        for camera in CAMERAS:
            for ccd in CCDS:
                name = chip_name(sector, camera, ccd)
                chunk = data_dir / "chunks" / f"{name}.parquet"
                final = out_dir / f"{name}.parquet"
                if final.is_file() or not chunk.is_file():
                    continue
                frame = pd.read_parquet(chunk)
                frame["TIC"] = frame["TIC"].astype(str)
                merged = frame.merge(
                    positions, on=["GAIADR3", "sector", "camera", "ccd"], how="left"
                )
                if len(merged) != len(frame):
                    raise AssertionError(f"{name}: row count changed on position merge")
                # A left-join miss means tess-point puts the star on a different chip
                # than its FITS header claims; those rows cannot be trusted for a
                # pixel-distance rule, so they are dropped.
                kept = merged[merged["DETECTOR_X"].notna()].reset_index(drop=True)
                report.append(
                    {
                        "chip": name,
                        "extracted": int(len(frame)),
                        "with_position": int(len(kept)),
                        "dropped_chip_mismatch": int(len(frame) - len(kept)),
                    }
                )
                if kept.empty:
                    print(f"{name}: no rows survived the position merge", flush=True)
                    continue
                part = final.with_name(final.name + ".part")
                kept.to_parquet(part)
                os.replace(part, final)
                print(
                    f"{name}: {len(kept):,} rows written "
                    f"({len(frame) - len(kept)} chip mismatches dropped)",
                    flush=True,
                )
    if report:
        frame = pd.DataFrame(report)
        frame.to_csv(data_dir / "position_merge_report.csv", index=False)
    return {"chips": len(report), "output": str(out_dir)}


# --------------------------------------------------------------------- coverage
def coverage_report(data_dir: Path) -> pd.DataFrame:
    """Detector span per chip -- the check that failed for the compact-patch fetch.

    A chip whose span cannot contain the outer ``[384, 768]`` px band is reported here
    rather than discovered later by an unexplained geometry failure.
    """
    out_dir = data_dir / "cross_sector_raw.parquet"
    rows = []
    for path in sorted(out_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["sector", "camera", "ccd", "DETECTOR_X", "DETECTOR_Y"])
        if frame.empty:
            continue
        x = frame["DETECTOR_X"].to_numpy(float)
        y = frame["DETECTOR_Y"].to_numpy(float)
        span_x, span_y = float(x.max() - x.min()), float(y.max() - y.min())
        rows.append(
            {
                "chip": path.stem,
                "stars": int(len(frame)),
                "x_min": float(x.min()), "x_max": float(x.max()),
                "y_min": float(y.min()), "y_max": float(y.max()),
                "span_x": span_x, "span_y": span_y,
                "max_separation": float(np.hypot(span_x, span_y)),
                "supports_outer_band": bool(np.hypot(span_x, span_y) >= 768.0),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(data_dir / "chip_coverage.csv", index=False)
    return frame


# -------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sectors", default="1,2,3,4,5")
    parser.add_argument(
        "--chips",
        default="all",
        help="camera-ccd pairs, e.g. '4-2' or '1-1,1-2'; 'all' means all sixteen",
    )
    parser.add_argument("--stars-per-chip", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--purge-fits",
        action="store_true",
        help="extract each chip right after downloading it and delete its FITS, "
             "so only one chip of raw files exists at a time (file-quota safety)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "index", "select", "download", "extract", "positions", "coverage"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sectors = [int(value) for value in str(args.sectors).split(",") if value.strip()]
    over = [s for s in sectors if s > MAX_PRIMARY_MISSION_SECTOR]
    if over:
        raise SystemExit(
            f"sectors {over} are 10-minute FFIs; they cannot share a 30-minute cadence "
            "grid with sectors 1-26"
        )
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    chips = all_chips(sectors, args.chips)
    stages = (
        ("index", "select", "download", "extract", "positions", "coverage")
        if args.stage == "all"
        else (args.stage,)
    )

    if "index" in stages:
        print(f"=== stage index: {len(sectors)} sectors ===", flush=True)
        for sector in sectors:
            build_sector_index(data_dir, sector)
    if "select" in stages:
        print(f"=== stage select: {len(chips)} chips ===", flush=True)
        for sector, camera, ccd in chips:
            select_chip(data_dir, sector, camera, ccd, args.stars_per_chip, args.seed)
    if "download" in stages:
        print(f"=== stage download: {len(chips)} chips ===", flush=True)
        for sector, camera, ccd in chips:
            download_chip(
                data_dir, sector, camera, ccd, args.workers,
                purge_fits=args.purge_fits,
            )
    if "extract" in stages:
        print(f"=== stage extract: {len(chips)} chips ===", flush=True)
        for sector, camera, ccd in chips:
            extract_chip(data_dir, sector, camera, ccd, args.workers)
    if "positions" in stages:
        print(f"=== stage positions: {len(sectors)} sectors ===", flush=True)
        merge_positions(data_dir, sectors)
    if "coverage" in stages:
        frame = coverage_report(data_dir)
        if frame.empty:
            print("no chips written yet", flush=True)
        else:
            bad = frame[~frame["supports_outer_band"]]
            print(
                f"coverage: {len(frame)} chips, {int(frame['stars'].sum()):,} stars; "
                f"median max-separation {frame['max_separation'].median():.0f} px",
                flush=True,
            )
            if len(bad):
                print(
                    f"WARNING: {len(bad)} chips cannot host the outer [384,768] px "
                    f"band:\n{bad[['chip', 'stars', 'max_separation']].to_string(index=False)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
