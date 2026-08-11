"""Verify that ``spread_geometry.SpreadPeerSelector`` reproduces the completed run.

The multi-chip scale-up applies the spread rule through a new module.  A rewritten
selector that quietly changes which peers are chosen would invalidate the comparison
with the completed Sector-1 / camera-4 / CCD-2 experiment, and would do it silently:
every downstream assertion in the trainer checks the *bands*, not the identities.

So this test rebuilds the original patch, replays the ported selector over the same
anchors and the same per-split pools, and requires the eight chosen peer TICs -- in
audited slot order -- to equal the run's ``peer_selection.csv`` for every anchor.

    python -m disentangle_attempt.test_spread_geometry_port
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from disentangle_attempt.spread_geometry import SpreadPeerSelector


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "outputs" / "local_s1_c4_ccd2_twodecoder_p64_i4_spread"
SPLITS = ("train", "val", "test")


def main() -> int:
    selection_path = RUN_DIR / "peer_selection.csv"
    config_path = RUN_DIR / "config.yaml"
    for path in (selection_path, config_path):
        if not path.is_file():
            print(f"SKIP: {path} is absent; nothing to verify against")
            return 0

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    audited = pd.read_csv(
        selection_path, dtype={"anchor_TIC": str, "peer_TIC": str}
    )
    audited["peer_order"] = audited["peer_order"].astype(int)

    from disentangle_attempt.spread_peers import build_spread_patch_from_config

    print("rebuilding the completed Sector-1/camera-4/CCD-2 patch ...", flush=True)
    patch = build_spread_patch_from_config(config, verbose=False)

    selector = SpreadPeerSelector(
        patch.det_x,
        patch.det_y,
        patch.tic,
        patch.tic_int,
        bands=config["peer_distance_bands_px"],
        peers_per_band=int(config["peers_per_distance_band"]),
        spacing_ladder=config["minimum_peer_separation_tiers_px"],
        outer_expansion_radii=config.get("outer_expansion_radii_px", ()),
        n_peers=int(config["n_peers"]),
    )

    target = tuple(int(v) for v in patch.target)
    checked = mismatched = 0
    failures: list[str] = []
    for split in SPLITS:
        pool = patch.split_pool[split][target]
        for anchor in patch.split_anchors[split]:
            anchor = int(anchor)
            anchor_tic = str(patch.tic[anchor])
            expected = (
                audited[
                    (audited["split"] == split)
                    & (audited["anchor_TIC"] == anchor_tic)
                ]
                .sort_values("peer_order")["peer_TIC"]
                .tolist()
            )
            if len(expected) != selector.n_peers:
                failures.append(f"{split}/{anchor_tic}: audit has {len(expected)} rows")
                mismatched += 1
                continue
            rows, _ = selector.select_for_anchor(anchor, pool)
            checked += 1
            if rows is None:
                failures.append(f"{split}/{anchor_tic}: port found no group")
                mismatched += 1
                continue
            actual = [str(patch.tic[row]) for row in rows]
            if actual != expected:
                mismatched += 1
                if len(failures) < 5:
                    failures.append(
                        f"{split}/{anchor_tic}:\n    audited={expected}\n    ported ={actual}"
                    )
        print(f"  {split}: {len(patch.split_anchors[split])} anchors replayed", flush=True)

    print(f"\nanchors compared: {checked}; mismatches: {mismatched}")
    if mismatched:
        print("FAIL: the ported selector does not reproduce the completed run")
        for line in failures:
            print("  " + line)
        return 1
    print("PASS: ported selector reproduces every audited peer group exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
