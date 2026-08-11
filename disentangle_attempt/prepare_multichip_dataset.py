"""Build and cache the multi-chip spread-peer dataset (the CPU stage).

Geometry selection is CPU-bound and training is GPU-bound, so on a cluster they are
separate jobs.  This stage loads every chip's curves, assigns the SHA-256 TIC split,
runs the two-band / 256-192-128 px spread rule for every candidate anchor, asserts the
data contract, and writes a cache the training stage memory-maps.

It refuses to train anything.  If the geometry is not feasible the failure surfaces
here, before a GPU is allocated.

    python -m disentangle_attempt.prepare_multichip_dataset \\
        --config disentangle_attempt/config_multichip_s1_s5.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml

from disentangle_attempt.multichip_spread_peers import (
    SPLITS,
    build_multichip_patch_from_config,
    save_cache,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config_multichip_s1_s5.yaml"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if a cache manifest already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.config.expanduser().resolve().open() as handle:
        config = yaml.safe_load(handle)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else resolve_repo_path(config["output_dir"])
    )
    cache_dir = output_dir / "dataset_cache"
    if (cache_dir / "manifest.json").is_file() and not args.force:
        print(f"cache already present at {cache_dir}; pass --force to rebuild")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet = (
        args.parquet.expanduser().resolve()
        if args.parquet is not None
        else resolve_repo_path(config["parquet"])
    )
    if not parquet.exists():
        raise FileNotFoundError(
            f"multi-chip parquet is missing: {parquet}\n"
            "run disentangle_attempt.multichip_acquire first"
        )
    if args.workers is not None:
        config["geometry_workers"] = int(args.workers)

    started = time.time()
    patch = build_multichip_patch_from_config(config, parquet_path=parquet, verbose=True)

    audit_dir = output_dir / "geometry"
    written = patch.save_audits(audit_dir)
    manifest = save_cache(patch, cache_dir)

    summary: dict[str, Any] = dict(manifest["geometry_summary"])
    summary["build_seconds"] = time.time() - started
    summary["cache_dir"] = str(cache_dir)
    summary["audit_files"] = written
    # The trainer refuses to start unless this says the geometry is feasible, so the
    # decision is recorded once, here, rather than re-derived later.
    summary["status"] = (
        "PASS"
        if all(len(patch.split_anchors[split]) > 0 for split in SPLITS)
        else "FAIL"
    )
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    print(f"\nstatus {summary['status']} in {summary['build_seconds']:.0f}s")
    print(f"cache:  {cache_dir}")
    print(f"audits: {audit_dir}")
    table = patch.per_chip_table()
    print("\nper-chip anchors:")
    print(table.to_string(index=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
