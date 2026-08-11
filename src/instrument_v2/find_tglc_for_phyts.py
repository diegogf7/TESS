"""Which TGLC parquet actually covers the PhyTS sector-14 cohort?

The MOMENT baseline matched only 65 of 2,409 PhyTS S14 rows against the default
``TGLC_PATH``, while the original physics A/B matched the full 2,409.  The join is on
``(GAIADR3, sector)``, so a low match rate means the parquet simply does not contain
those Gaia sources -- the A/B must have been pointed at a different file.

This scans candidate parquets and reports, for each, how many of the PhyTS S14 Gaia
ids it covers.  It reads ONLY the id columns, so it is safe on a login node: the
full-file read that gets sessions killed comes from pulling the light-curve arrays.

    python -m src.instrument_v2.find_tglc_for_phyts
    TGLC_SEARCH_DIRS=/path/a,/path/b python -m src.instrument_v2.find_tglc_for_phyts
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.instrument_v2.eval_phyts_instrument_ab import DATA_PATH as PHYTS_PATH

SEARCH_DIRS = os.environ.get(
    "TGLC_SEARCH_DIRS",
    ",".join(
        [
            "/orcd/scratch/orcd/006/diegogon/tglc_primary",
            "/orcd/scratch/orcd/006/diegogon",
            "/orcd/scratch/orcd/006/diegogon/phyts",
        ]
    ),
).split(",")


def phyts_gaia_ids() -> set[int]:
    phyts = pd.read_parquet(PHYTS_PATH)
    phyts = phyts[phyts["sector"] == 14]
    column = next(
        (c for c in ("GaiaID", "gaiaid", "GAIADR3", "GAIADR2", "gaia_id") if c in phyts.columns),
        None,
    )
    if column is None:
        raise RuntimeError(f"no Gaia column in {PHYTS_PATH}: {list(phyts.columns)}")
    ids = pd.to_numeric(phyts[column], errors="coerce").dropna().astype(np.int64)
    print(f"PhyTS s14: {len(phyts)} rows, {ids.nunique()} unique Gaia ids "
          f"(column {column!r})", flush=True)
    return set(ids.tolist())


def main() -> None:
    wanted = phyts_gaia_ids()
    candidates: list[str] = []
    for directory in SEARCH_DIRS:
        directory = directory.strip()
        if not directory or not os.path.isdir(directory):
            continue
        candidates.extend(glob.glob(os.path.join(directory, "*.parquet")))
        candidates.extend(glob.glob(os.path.join(directory, "*", "*.parquet")))
    candidates = sorted(set(candidates))
    if not candidates:
        raise SystemExit(f"no parquet files found under {SEARCH_DIRS}")

    print(f"\nscanning {len(candidates)} parquet files\n", flush=True)
    rows = []
    for path in candidates:
        try:
            names = set(pq.read_schema(path).names)
            gaia_col = next((c for c in ("GAIADR3", "GaiaID", "gaiaid", "gaia_id") if c in names), None)
            if gaia_col is None:
                continue
            columns = [gaia_col] + (["sector"] if "sector" in names else [])
            frame = pd.read_parquet(path, columns=columns)
            ids = pd.to_numeric(frame[gaia_col], errors="coerce").dropna().astype(np.int64)
            if "sector" in frame.columns:
                s14 = ids[frame.loc[ids.index, "sector"] == 14]
            else:
                s14 = ids
            covered = len(wanted & set(s14.tolist()))
            rows.append(
                {
                    "path": path,
                    "rows": int(len(frame)),
                    "s14_rows": int(len(s14)),
                    "phyts_covered": covered,
                    "coverage_pct": round(100.0 * covered / max(len(wanted), 1), 1),
                    "has_flux": bool({"aperture_flux", "flux"} & names),
                    "has_flags": bool({"TESS_flags", "TGLC_flags"} <= names),
                }
            )
            print(
                f"  {covered:5d}/{len(wanted)} ({rows[-1]['coverage_pct']:5.1f}%)  "
                f"flux={rows[-1]['has_flux']} flags={rows[-1]['has_flags']}  {path}",
                flush=True,
            )
        except Exception as exc:  # a malformed or unrelated parquet must not stop the scan
            print(f"  skipped {path}: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        raise SystemExit("no parquet with a Gaia id column was readable")
    table = pd.DataFrame(rows).sort_values("phyts_covered", ascending=False)
    usable = table[table["has_flux"] & table["has_flags"]]
    print("\n=== best coverage (with flux AND flags, i.e. usable for the eval) ===")
    print(
        (usable if len(usable) else table)
        .head(5)[["phyts_covered", "coverage_pct", "s14_rows", "has_flux", "has_flags", "path"]]
        .to_string(index=False)
    )
    if len(usable):
        best = usable.iloc[0]
        print(
            f"\nrun the baseline with:\n"
            f"    TGLC_PATH={best['path']} \\\n"
            f"    MOMENT_ARM=both python -m src.instrument_v2.eval_moment_phyts_baseline"
        )


if __name__ == "__main__":
    main()
