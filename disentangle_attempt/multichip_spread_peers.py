"""Many-chip spread-peer patch: the single-chip experiment, applied per chip.

The completed experiment trained one chip (Sector 1 / camera 4 / CCD 2) with 320/40/40
anchors.  This module builds the same thing over every sector/camera/CCD chip in a
multi-chip parquet, so the scaled run is that experiment repeated across chips rather
than a different experiment.

What is deliberately identical:

* the flux contract -- raw TGLC ``aperture_flux`` through
  :func:`disentangle_attempt.preprocess.preprocess_curve`, strict zero-flag validity,
  median/MAD normalization from valid cadences only;
* one absolute cadence grid per sector, curves placed by exact cadence number;
* the peer rule -- :class:`disentangle_attempt.spread_geometry.SpreadPeerSelector`,
  verified against the completed run by ``test_spread_geometry_port.py``;
* peers drawn only from the anchor's own chip and own split.

What necessarily differs:

* the TIC split is the documented SHA-256 80/10/10 hash
  (:func:`disentangle_attempt.spread_peers.stable_peer_split`) applied to every TIC in
  the data set, so a TIC observed in several sectors lands in one split globally and
  cannot leak between splits through a second sector;
* curves are loaded and gridded one chip at a time, and only ``X`` (float32) and ``M``
  (bool) are retained.  Keeping ``CrossSectorPatch``'s int64 flag arrays for 240k stars
  would cost ~4 GB on their own and the whole raw frame ~15 GB.  Flag policy is instead
  asserted per chip at load time and then summarized.

Detector coordinates are used only to select peers.  They are never model inputs,
targets, or a constructed correction curve.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from disentangle_attempt.preprocess import preprocess_curve
from disentangle_attempt.spread_geometry import (
    SPACING_LADDER,
    SPREAD_BANDS,
    SpreadPeerSelector,
)
from disentangle_attempt.spread_peers import canonical_numeric_tic, stable_peer_split


SPLITS = ("train", "val", "test")
CURVE_LENGTH = 1024


def anchor_rank(tic: str, sector: int, camera: int, ccd: int, seed: int) -> bytes:
    """Deterministic anchor ordering that does not depend on catalogue order."""
    key = f"{seed}:anchor:{sector}:{camera}:{ccd}:{tic}".encode("ascii")
    return hashlib.sha256(key).digest()


def grid_from_counts(counts: dict[int, int], length: int) -> np.ndarray:
    """The `length` consecutive cadence numbers covering the most observations.

    Same rule as :func:`disentangle_attempt.dataset.build_cadence_grid`, but fed by an
    accumulated histogram so no sector's cadence arrays are ever all in memory at once.
    """
    if not counts:
        raise ValueError("no cadences to build a grid from")
    keys = np.fromiter(sorted(counts), dtype=np.int64)
    low, high = int(keys[0]), int(keys[-1])
    if high - low + 1 <= length:
        return np.arange(low, low + length, dtype=np.int64)
    dense = np.zeros(high - low + 1, dtype=np.int64)
    dense[keys - low] = np.fromiter((counts[int(k)] for k in keys), dtype=np.int64)
    window = np.convolve(dense, np.ones(length, dtype=np.int64), mode="valid")
    start = low + int(np.argmax(window))
    return np.arange(start, start + length, dtype=np.int64)


# ----------------------------------------------------------- parallel geometry
_WORKER: dict[str, Any] = {}


def _init_worker(
    det_x: np.ndarray,
    det_y: np.ndarray,
    tic: np.ndarray,
    tic_int: np.ndarray,
    rule: dict[str, Any],
) -> None:
    _WORKER["selector"] = SpreadPeerSelector(det_x, det_y, tic, tic_int, **rule)


def _select_chip_split(
    task: tuple[tuple[int, int, int], str, np.ndarray, np.ndarray],
) -> tuple[tuple[int, int, int], str, list[dict[str, Any]]]:
    """Run the spread rule for every anchor of one (chip, split); pure and picklable."""
    chip, split, anchors, pool = task
    selector: SpreadPeerSelector = _WORKER["selector"]
    results: list[dict[str, Any]] = []
    for anchor in anchors:
        rows, attempts = selector.select_for_anchor(int(anchor), pool)
        results.append(
            {"anchor_row": int(anchor), "rows": rows, "attempts": attempts}
        )
    return chip, split, results


class MultiChipSpreadPatch:
    """Gridded curves, SHA-256 TIC splits and per-chip spread-peer tables."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        curve_length: int = CURVE_LENGTH,
        n_peers: int = 8,
        min_valid_fraction: float = 0.5,
        split_seed: int = 42,
        anchors_per_chip: int = 400,
        min_chip_pool: int = 64,
        bands: Sequence[Sequence[float]] = SPREAD_BANDS,
        peers_per_band: int = 4,
        spacing_ladder: Sequence[float] = SPACING_LADDER,
        outer_expansion_radii: Sequence[float] = (),
        exact_search_node_budget: int = 1_000_000,
        sectors: Iterable[int] | None = None,
        workers: int | None = None,
        verbose: bool = True,
    ):
        self.parquet_path = Path(parquet_path).expanduser().resolve()
        self.curve_length = int(curve_length)
        self.n_peers = int(n_peers)
        self.min_valid_fraction = float(min_valid_fraction)
        self.min_valid = int(self.min_valid_fraction * self.curve_length)
        self.split_seed = int(split_seed)
        self.anchors_per_chip = int(anchors_per_chip)
        self.min_chip_pool = int(min_chip_pool)
        self.peers_per_band = int(peers_per_band)
        self.verbose = bool(verbose)
        self._rule = {
            "bands": [list(band) for band in bands],
            "peers_per_band": int(peers_per_band),
            "spacing_ladder": list(spacing_ladder),
            "outer_expansion_radii": list(outer_expansion_radii),
            "n_peers": self.n_peers,
            "exact_search_node_budget": int(exact_search_node_budget),
        }
        self.workers = int(workers) if workers else max(1, (os.cpu_count() or 2) - 1)

        self._chip_files = self._discover_chip_files(sectors)
        self._load_curves()
        self._assign_splits()
        self._build_pools_and_anchors()
        self._build_peer_tables()
        self._assert_contract()
        if self.verbose:
            self.report()

    # ------------------------------------------------------------------ loading
    def _discover_chip_files(self, sectors: Iterable[int] | None) -> list[Path]:
        if self.parquet_path.is_dir():
            files = sorted(self.parquet_path.glob("*.parquet"))
        elif self.parquet_path.is_file():
            files = [self.parquet_path]
        else:
            raise FileNotFoundError(f"no parquet at {self.parquet_path}")
        if not files:
            raise FileNotFoundError(f"parquet directory is empty: {self.parquet_path}")
        if sectors is not None:
            wanted = {int(value) for value in sectors}
            kept = []
            for path in files:
                # Chip files are named s{sector:04d}_cam{c}_ccd{d}.parquet.
                stem = path.stem
                if stem.startswith("s") and "_" in stem:
                    try:
                        if int(stem[1:5]) not in wanted:
                            continue
                    except ValueError:
                        pass
                kept.append(path)
            files = kept
            if not files:
                raise FileNotFoundError(f"no chip files for sectors {sorted(wanted)}")
        return files

    def _load_curves(self) -> None:
        started = time.time()
        # Pass 1: accumulate a per-sector cadence histogram without holding any curves.
        cadence_counts: dict[int, dict[int, int]] = {}
        for path in self._chip_files:
            frame = pd.read_parquet(path, columns=["sector", "cadence_num"])
            for sector, group in frame.groupby("sector"):
                target = cadence_counts.setdefault(int(sector), {})
                stacked = np.concatenate(
                    [np.asarray(a, dtype=np.int64) for a in group["cadence_num"]]
                )
                values, counts = np.unique(stacked, return_counts=True)
                for value, count in zip(values.tolist(), counts.tolist()):
                    target[value] = target.get(value, 0) + count
            del frame
        self.grids = {
            sector: grid_from_counts(counts, self.curve_length)
            for sector, counts in cadence_counts.items()
        }
        if self.verbose:
            print(
                f"cadence grids built for sectors {sorted(self.grids)} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

        # Pass 2: grid one chip at a time and keep only what a model or the peer rule
        # needs.  Flag policy is asserted here, then the flag arrays are discarded.
        x_blocks: list[np.ndarray] = []
        m_blocks: list[np.ndarray] = []
        meta_blocks: list[pd.DataFrame] = []
        flag_rows: list[dict[str, Any]] = []
        for path in self._chip_files:
            frame = pd.read_parquet(path)
            frame["TIC"] = frame["TIC"].astype(str)
            frame = frame.drop_duplicates(["TIC", "sector"]).reset_index(drop=True)
            frame = frame[
                frame["DETECTOR_X"].notna() & frame["DETECTOR_Y"].notna()
            ].reset_index(drop=True)
            if frame.empty:
                del frame
                continue
            count = len(frame)
            X = np.zeros((count, self.curve_length), dtype=np.float32)
            M = np.zeros((count, self.curve_length), dtype=bool)
            removed = 0
            total = 0
            for index in range(count):
                row = frame.iloc[index]
                result = preprocess_curve(
                    row["cadence_num"],
                    row.get("time"),
                    row["flux"],
                    row.get("TESS_flags"),
                    row.get("TGLC_flags"),
                    self.grids[int(row["sector"])],
                )
                X[index] = result.curve
                M[index] = result.valid
                # Strict zero-flag policy, enforced rather than assumed.  Nothing
                # flagged may survive as a valid, model-visible cadence.
                if bool((result.valid & result.flagged).any()):
                    raise AssertionError(
                        f"{path.name} row {index}: a flagged cadence survived as valid"
                    )
                removed += int(result.flagged.sum())
                total += int(self.curve_length)
            x_blocks.append(X)
            m_blocks.append(M)
            meta_blocks.append(
                frame[
                    ["TIC", "sector", "camera", "ccd", "DETECTOR_X", "DETECTOR_Y"]
                ].copy()
            )
            flag_rows.append(
                {
                    "chip_file": path.name,
                    "curves": count,
                    "gridded_cadences": total,
                    "flagged_cadences": removed,
                    "valid_cadences": int(M.sum()),
                }
            )
            del frame
        if not x_blocks:
            raise RuntimeError("no curves survived loading")

        self.X = np.concatenate(x_blocks, axis=0)
        self.M = np.concatenate(m_blocks, axis=0)
        del x_blocks, m_blocks
        meta = pd.concat(meta_blocks, ignore_index=True)
        del meta_blocks
        self.flag_report = pd.DataFrame(flag_rows)

        self.tic = meta["TIC"].astype(str).to_numpy()
        self.tic_int = meta["TIC"].astype(np.int64).to_numpy()
        self.sector = meta["sector"].astype(int).to_numpy()
        self.camera = meta["camera"].astype(int).to_numpy()
        self.ccd = meta["ccd"].astype(int).to_numpy()
        self.det_x = meta["DETECTOR_X"].astype(float).to_numpy()
        self.det_y = meta["DETECTOR_Y"].astype(float).to_numpy()
        self.n_valid = self.M.sum(axis=1).astype(np.int64)

        self.chips = sorted(
            {
                (int(s), int(c), int(d))
                for s, c, d in zip(self.sector, self.camera, self.ccd)
            }
        )
        self.chip_of_row = {}
        self.rows_by_chip: dict[tuple[int, int, int], np.ndarray] = {}
        for chip in self.chips:
            rows = np.flatnonzero(
                (self.sector == chip[0])
                & (self.camera == chip[1])
                & (self.ccd == chip[2])
            ).astype(np.int64)
            self.rows_by_chip[chip] = rows
            for row in rows:
                self.chip_of_row[int(row)] = chip
        if self.verbose:
            print(
                f"loaded {len(self.tic):,} curves over {len(self.chips)} chips "
                f"({self.X.nbytes / 1e9:.2f} GB flux + {self.M.nbytes / 1e9:.2f} GB "
                f"validity) in {time.time() - started:.0f}s",
                flush=True,
            )

    # ------------------------------------------------------------------- splits
    def _assign_splits(self) -> None:
        """Documented SHA-256 80/10/10 over every TIC, independent of catalogue order."""
        self.split_of_tic: dict[str, str] = {}
        records: list[dict[str, Any]] = []
        for tic in sorted(set(self.tic), key=int):
            assignment = stable_peer_split(canonical_numeric_tic(tic), self.split_seed)
            self.split_of_tic[str(tic)] = assignment["split"]
            records.append(assignment)
        self.split_assignment = pd.DataFrame(records)
        self.split_tics = {
            split: {tic for tic, value in self.split_of_tic.items() if value == split}
            for split in SPLITS
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if self.split_tics[left] & self.split_tics[right]:
                raise AssertionError(f"{left}/{right} TIC leakage")
        self.row_split = np.asarray(
            [self.split_of_tic[str(value)] for value in self.tic]
        )

    # -------------------------------------------------------- pools and anchors
    def _build_pools_and_anchors(self) -> None:
        eligible = self.n_valid >= self.min_valid
        self.eligible_mask = eligible
        self.split_pool: dict[str, dict[tuple[int, int, int], np.ndarray]] = {
            split: {} for split in SPLITS
        }
        candidate_anchors: dict[str, list[int]] = {split: [] for split in SPLITS}
        self.thin_chip_splits: list[dict[str, Any]] = []

        for chip in self.chips:
            rows = self.rows_by_chip[chip]
            usable = rows[eligible[rows]]
            for split in SPLITS:
                pool = usable[self.row_split[usable] == split]
                if len(pool) < self.min_chip_pool:
                    # Too thin to host a spread group; recorded, not silently dropped.
                    self.thin_chip_splits.append(
                        {
                            "chip": f"s{chip[0]:04d}_cam{chip[1]}_ccd{chip[2]}",
                            "split": split,
                            "pool": int(len(pool)),
                            "required": self.min_chip_pool,
                        }
                    )
                    continue
                self.split_pool[split][chip] = pool
            # Anchor candidates are chosen per chip by a stable hash, before any
            # geometry is attempted, so the anchor set does not depend on which
            # anchors happen to succeed.
            ordered = sorted(
                (int(row) for row in usable),
                key=lambda row: anchor_rank(
                    str(self.tic[row]), chip[0], chip[1], chip[2], self.split_seed
                ),
            )
            taken = 0
            for row in ordered:
                if taken >= self.anchors_per_chip:
                    break
                split = self.row_split[row]
                if chip not in self.split_pool[split]:
                    continue
                candidate_anchors[split].append(row)
                taken += 1
        self.candidate_anchors = {
            split: np.asarray(sorted(rows), dtype=np.int64)
            for split, rows in candidate_anchors.items()
        }

    # -------------------------------------------------------------- peer tables
    def _build_peer_tables(self) -> None:
        tasks: list[tuple[tuple[int, int, int], str, np.ndarray, np.ndarray]] = []
        for split in SPLITS:
            anchors = self.candidate_anchors[split]
            for chip in self.chips:
                if chip not in self.split_pool[split]:
                    continue
                chip_anchors = np.asarray(
                    [row for row in anchors if self.chip_of_row[int(row)] == chip],
                    dtype=np.int64,
                )
                if len(chip_anchors):
                    tasks.append((chip, split, chip_anchors, self.split_pool[split][chip]))
        if not tasks:
            raise RuntimeError("no (chip, split) has both anchors and a peer pool")

        started = time.time()
        total_anchors = sum(len(task[2]) for task in tasks)
        if self.verbose:
            print(
                f"selecting spread peers for {total_anchors:,} candidate anchors over "
                f"{len(tasks)} chip/split groups with {self.workers} workers",
                flush=True,
            )
        results: list[tuple[tuple[int, int, int], str, list[dict[str, Any]]]] = []
        initargs = (self.det_x, self.det_y, self.tic, self.tic_int, self._rule)
        if self.workers > 1:
            with ProcessPoolExecutor(
                max_workers=self.workers, initializer=_init_worker, initargs=initargs
            ) as pool:
                for done, item in enumerate(pool.map(_select_chip_split, tasks), 1):
                    results.append(item)
                    if self.verbose and done % 20 == 0:
                        print(
                            f"  {done}/{len(tasks)} groups "
                            f"({time.time() - started:.0f}s)",
                            flush=True,
                        )
        else:
            _init_worker(*initargs)
            results = [_select_chip_split(task) for task in tasks]

        selected_rows: dict[int, list[int]] = {}
        attempt_records: list[dict[str, Any]] = []
        self.excluded_anchors: dict[str, list[int]] = {split: [] for split in SPLITS}
        for chip, split, group in results:
            for item in group:
                anchor = int(item["anchor_row"])
                for attempt in item["attempts"]:
                    attempt = dict(attempt)
                    attempt["split"] = split
                    attempt["chip"] = f"s{chip[0]:04d}_cam{chip[1]}_ccd{chip[2]}"
                    attempt_records.append(attempt)
                if item["rows"] is None:
                    self.excluded_anchors[split].append(anchor)
                else:
                    selected_rows[anchor] = [int(value) for value in item["rows"]]
        self.peer_selection_attempts = pd.DataFrame(attempt_records)

        self.split_anchors: dict[str, np.ndarray] = {}
        self.peers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        selection_records: list[dict[str, Any]] = []
        for split in SPLITS:
            kept = np.asarray(
                sorted(row for row in self.candidate_anchors[split] if int(row) in selected_rows),
                dtype=np.int64,
            )
            rows_out = np.full((len(kept), self.n_peers), -1, dtype=np.int64)
            distances_out = np.full((len(kept), self.n_peers), np.nan, dtype=np.float32)
            for index, anchor in enumerate(kept):
                peers = np.asarray(selected_rows[int(anchor)], dtype=np.int64)
                distance = np.hypot(
                    self.det_x[peers] - self.det_x[int(anchor)],
                    self.det_y[peers] - self.det_y[int(anchor)],
                )
                rows_out[index] = peers
                distances_out[index] = distance.astype(np.float32)
                chip = self.chip_of_row[int(anchor)]
                xy = np.column_stack([self.det_x[peers], self.det_y[peers]])
                pair = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
                group_minimum = float(pair[np.triu_indices(self.n_peers, 1)].min())
                for order, (peer, anchor_distance) in enumerate(
                    zip(peers, distance), start=1
                ):
                    selection_records.append(
                        {
                            "split": split,
                            "chip": f"s{chip[0]:04d}_cam{chip[1]}_ccd{chip[2]}",
                            "anchor_row": int(anchor),
                            "anchor_TIC": str(self.tic[int(anchor)]),
                            "anchor_x": float(self.det_x[int(anchor)]),
                            "anchor_y": float(self.det_y[int(anchor)]),
                            "peer_order": order,
                            "peer_row": int(peer),
                            "peer_TIC": str(self.tic[int(peer)]),
                            "peer_x": float(self.det_x[int(peer)]),
                            "peer_y": float(self.det_y[int(peer)]),
                            "band_index": 0 if order <= self.peers_per_band else 1,
                            "anchor_distance": float(anchor_distance),
                            "group_minimum_peer_separation": group_minimum,
                        }
                    )
            self.split_anchors[split] = kept
            self.peers[split] = (rows_out, distances_out)
        self.peer_selection_audit = pd.DataFrame(selection_records)
        if self.verbose:
            print(
                f"spread-peer selection finished in {time.time() - started:.0f}s",
                flush=True,
            )

    # --------------------------------------------------------------- assertions
    def _assert_contract(self) -> None:
        for split in SPLITS:
            rows, distances = self.peers[split]
            anchors = self.split_anchors[split]
            if len(anchors) == 0:
                # Say why, not just that.  The usual cause is a per-chip pool too thin
                # for this split, which happens long before the geometry is consulted.
                thin = [
                    item for item in self.thin_chip_splits if item["split"] == split
                ]
                detail = [
                    f"split {split!r} kept no anchors",
                    f"  candidate anchors offered : {len(self.candidate_anchors[split])}",
                    f"  chips with a usable pool  : {len(self.split_pool[split])} of {len(self.chips)}",
                    f"  chips whose pool was thin : {len(thin)} (need >= {self.min_chip_pool} stars)",
                ]
                if thin:
                    worst = sorted(thin, key=lambda item: -item["pool"])[:5]
                    detail.append(
                        "  largest thin pools        : "
                        + ", ".join(f"{i['chip']}={i['pool']}" for i in worst)
                    )
                    detail.append(
                        f"  the {split} split is ~{'80' if split == 'train' else '10'}% of each "
                        "chip's stars, so raise --stars-per-chip or lower min_chip_pool"
                    )
                raise RuntimeError("\n".join(detail))
            if rows.shape != (len(anchors), self.n_peers):
                raise AssertionError(f"{split} peer table shape is wrong")
            if (rows < 0).any() or not np.isfinite(distances).all():
                raise AssertionError(f"{split} peer table contains placeholders")
            for index, anchor in enumerate(anchors):
                anchor = int(anchor)
                peers = rows[index]
                if len(set(peers.tolist())) != self.n_peers:
                    raise AssertionError(f"{split} anchor {anchor}: peers not unique")
                if (self.tic[peers] == self.tic[anchor]).any():
                    raise AssertionError(f"{split} anchor {anchor}: anchor TIC as peer")
                if not (
                    (self.sector[peers] == self.sector[anchor]).all()
                    and (self.camera[peers] == self.camera[anchor]).all()
                    and (self.ccd[peers] == self.ccd[anchor]).all()
                ):
                    raise AssertionError(f"{split} anchor {anchor}: peer left the chip")
                if {self.split_of_tic[str(t)] for t in self.tic[peers]} != {split}:
                    raise AssertionError(f"{split} anchor {anchor}: cross-split peer")
                inner = distances[index][: self.peers_per_band]
                outer = distances[index][self.peers_per_band :]
                if not ((inner >= 128.0) & (inner < 384.0)).all():
                    raise AssertionError(f"{split} anchor {anchor}: inner band violated")
                if not (outer >= 384.0).all():
                    raise AssertionError(f"{split} anchor {anchor}: outer band violated")
        # Selected peers must not leak across splits either.
        selected = {
            split: set(self.tic[self.peers[split][0].reshape(-1)].tolist())
            for split in SPLITS
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if selected[left] & selected[right]:
                raise AssertionError(f"selected peer TICs leak between {left} and {right}")

    # ------------------------------------------------------------------ reports
    def geometry_summary(self) -> dict[str, Any]:
        audit = self.peer_selection_audit
        summary: dict[str, Any] = {
            "chips": len(self.chips),
            "curves": int(len(self.tic)),
            "curve_length": self.curve_length,
            "min_valid_cadences": self.min_valid,
            "anchors_per_chip_cap": self.anchors_per_chip,
            "split_seed": self.split_seed,
            "split_method": "sha256_tic_80_10_10",
            "bands_px": self._rule["bands"],
            "peers_per_band": self.peers_per_band,
            "spacing_ladder_px": self._rule["spacing_ladder"],
            "selected_anchor_counts": {
                split: int(len(self.split_anchors[split])) for split in SPLITS
            },
            "candidate_anchor_counts": {
                split: int(len(self.candidate_anchors[split])) for split in SPLITS
            },
            "excluded_anchor_counts": {
                split: int(len(self.excluded_anchors[split])) for split in SPLITS
            },
            "peer_pool_counts": {
                split: int(sum(len(pool) for pool in self.split_pool[split].values()))
                for split in SPLITS
            },
            "thin_chip_splits": len(self.thin_chip_splits),
            "unique_selected_peer_TICs": int(audit["peer_TIC"].nunique())
            if not audit.empty
            else 0,
        }
        if not self.peer_selection_attempts.empty:
            terminal = (
                self.peer_selection_attempts.sort_values(
                    ["split", "anchor_row", "spacing_fallback_level"], kind="stable"
                )
                .groupby(["split", "anchor_row"], sort=False)
                .tail(1)
            )
            summary["terminal_spacing_tier_counts"] = {
                str(int(tier)): int(count)
                for tier, count in terminal["spacing_tier"].value_counts().sort_index().items()
            }
            summary["terminal_expansion_count"] = int(terminal["expansion_used"].sum())
        if not audit.empty:
            distance = audit["anchor_distance"].to_numpy(float)
            group_min = audit.drop_duplicates(["split", "anchor_row"])[
                "group_minimum_peer_separation"
            ].to_numpy(float)
            summary["anchor_to_peer_distance_px"] = {
                "minimum": float(distance.min()),
                "median": float(np.median(distance)),
                "maximum": float(distance.max()),
            }
            summary["group_minimum_peer_separation_px"] = {
                "minimum": float(group_min.min()),
                "median": float(np.median(group_min)),
                "maximum": float(group_min.max()),
            }
        return summary

    def per_chip_table(self) -> pd.DataFrame:
        rows = []
        for chip in self.chips:
            name = f"s{chip[0]:04d}_cam{chip[1]}_ccd{chip[2]}"
            record: dict[str, Any] = {"chip": name, "curves": int(len(self.rows_by_chip[chip]))}
            for split in SPLITS:
                pool = self.split_pool[split].get(chip)
                record[f"{split}_pool"] = int(len(pool)) if pool is not None else 0
                record[f"{split}_anchors"] = int(
                    sum(
                        1
                        for row in self.split_anchors[split]
                        if self.chip_of_row[int(row)] == chip
                    )
                )
            rows.append(record)
        return pd.DataFrame(rows)

    def save_audits(self, output_dir: str | Path) -> dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for name, frame in (
            ("peer_selection.csv", self.peer_selection_audit),
            ("peer_selection_attempts.csv", self.peer_selection_attempts),
            ("split_assignment.csv", self.split_assignment),
            ("per_chip_summary.csv", self.per_chip_table()),
            ("flag_report.csv", self.flag_report),
            ("thin_chip_splits.csv", pd.DataFrame(self.thin_chip_splits)),
        ):
            path = output_dir / name
            frame.to_csv(path, index=False)
            written[name] = str(path)
        return written

    def report(self) -> None:
        summary = self.geometry_summary()
        print(
            f"multi-chip spread patch: {summary['chips']} chips, "
            f"{summary['curves']:,} curves",
            flush=True,
        )
        for split in SPLITS:
            print(
                f"  {split}: {summary['selected_anchor_counts'][split]:,} anchors "
                f"(of {summary['candidate_anchor_counts'][split]:,} candidates; "
                f"{summary['excluded_anchor_counts'][split]:,} without a feasible "
                f"group), peer pool {summary['peer_pool_counts'][split]:,}",
                flush=True,
            )
        if "terminal_spacing_tier_counts" in summary:
            print(
                f"  spacing tiers used: {summary['terminal_spacing_tier_counts']}; "
                f"outer expansions: {summary['terminal_expansion_count']}",
                flush=True,
            )
        if "anchor_to_peer_distance_px" in summary:
            distance = summary["anchor_to_peer_distance_px"]
            group = summary["group_minimum_peer_separation_px"]
            print(
                f"  anchor-peer distance px: min {distance['minimum']:.1f}, "
                f"median {distance['median']:.1f}, max {distance['maximum']:.1f}",
                flush=True,
            )
            print(
                f"  group minimum peer separation px: min {group['minimum']:.1f}, "
                f"median {group['median']:.1f}",
                flush=True,
            )
        if self.thin_chip_splits:
            print(
                f"  {len(self.thin_chip_splits)} chip/split pools were too thin "
                f"(< {self.min_chip_pool}) and contributed no anchors",
                flush=True,
            )


CACHE_ARRAYS = (
    "X", "M", "tic_int", "sector", "camera", "ccd", "det_x", "det_y", "n_valid"
)


def save_cache(patch: "MultiChipSpreadPatch", cache_dir: str | Path) -> dict[str, Any]:
    """Persist the built patch so training need not redo loading and geometry.

    The geometry stage is CPU-bound and the training stage is GPU-bound, so on a
    cluster they are separate jobs.  Arrays are written individually (not as one npz)
    so the trainer can memory-map the ~1 GB flux array instead of reading it whole.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in CACHE_ARRAYS:
        np.save(cache_dir / f"{name}.npy", getattr(patch, name))
    np.save(cache_dir / "tic.npy", patch.tic.astype("U20"))
    for split in SPLITS:
        rows, distances = patch.peers[split]
        np.save(cache_dir / f"anchors_{split}.npy", patch.split_anchors[split])
        np.save(cache_dir / f"peer_rows_{split}.npy", rows)
        np.save(cache_dir / f"peer_distances_{split}.npy", distances)
    manifest = {
        "format": "multichip_spread_cache_v1",
        "parquet_path": str(patch.parquet_path),
        "curve_length": patch.curve_length,
        "n_peers": patch.n_peers,
        "split_seed": patch.split_seed,
        "min_valid": patch.min_valid,
        "anchors_per_chip": patch.anchors_per_chip,
        "chips": [list(chip) for chip in patch.chips],
        "rule": patch._rule,
        "geometry_summary": patch.geometry_summary(),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return manifest


class CachedMultiChipPatch:
    """Read-only patch rebuilt from :func:`save_cache`, for the training stage.

    Exposes exactly the attributes ``CrossSectorAnchorDataset`` and the trainer's
    held-out inference touch, so the training code cannot tell it apart from a freshly
    built patch.
    """

    def __init__(self, cache_dir: str | Path, mmap: bool = True):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no dataset cache at {self.cache_dir}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "multichip_spread_cache_v1":
            raise ValueError(f"unexpected cache format: {self.manifest.get('format')}")
        mode = "r" if mmap else None
        for name in CACHE_ARRAYS:
            setattr(self, name, np.load(self.cache_dir / f"{name}.npy", mmap_mode=mode))
        self.tic = np.load(self.cache_dir / "tic.npy")
        self.curve_length = int(self.manifest["curve_length"])
        self.n_peers = int(self.manifest["n_peers"])
        self.chips = [tuple(int(v) for v in chip) for chip in self.manifest["chips"]]
        self.split_anchors = {
            split: np.load(self.cache_dir / f"anchors_{split}.npy") for split in SPLITS
        }
        self.peers = {
            split: (
                np.load(self.cache_dir / f"peer_rows_{split}.npy"),
                np.load(self.cache_dir / f"peer_distances_{split}.npy"),
            )
            for split in SPLITS
        }
        self.chip_of_row = {}
        for chip in self.chips:
            rows = np.flatnonzero(
                (self.sector == chip[0])
                & (self.camera == chip[1])
                & (self.ccd == chip[2])
            )
            for row in rows:
                self.chip_of_row[int(row)] = chip

    def chip_name_of_row(self, row: int) -> str:
        sector, camera, ccd = self.chip_of_row[int(row)]
        return f"s{sector:04d}_cam{camera}_ccd{ccd}"


def build_multichip_patch_from_config(
    config: dict[str, Any], parquet_path: str | Path | None = None, verbose: bool = True
) -> MultiChipSpreadPatch:
    return MultiChipSpreadPatch(
        parquet_path or config["parquet"],
        curve_length=int(config.get("curve_length", CURVE_LENGTH)),
        n_peers=int(config.get("n_peers", 8)),
        min_valid_fraction=float(config.get("min_valid_fraction", 0.5)),
        split_seed=int(config.get("seed", 42)),
        anchors_per_chip=int(config.get("anchors_per_chip", 400)),
        min_chip_pool=int(config.get("min_chip_pool", 64)),
        bands=config.get("peer_distance_bands_px", SPREAD_BANDS),
        peers_per_band=int(config.get("peers_per_distance_band", 4)),
        spacing_ladder=config.get("minimum_peer_separation_tiers_px", SPACING_LADDER),
        outer_expansion_radii=config.get("outer_expansion_radii_px", ()),
        exact_search_node_budget=int(config.get("exact_search_node_budget", 1_000_000)),
        sectors=config.get("sectors"),
        workers=config.get("geometry_workers"),
        verbose=verbose,
    )


__all__ = [
    "MultiChipSpreadPatch",
    "build_multichip_patch_from_config",
    "grid_from_counts",
    "anchor_rank",
    "SPLITS",
]
