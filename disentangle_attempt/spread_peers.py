"""Fixed-anchor, leakage-safe spread-peer patch for the two-decoder experiment.

This module deliberately keeps the legacy Sector 1 / camera 4 / CCD 2 anchor
construction separate from a supplemental, peer-only catalog.  The original TIC
assignment and 320/40/40 anchors are reconstructed from the original parquet first;
only genuinely new TICs are assigned with a stable per-TIC hash.  Adding supplemental
rows therefore cannot reshuffle an old TIC, change an anchor, or change the cadence
grid.

Detector coordinates are used only for peer selection.  They are never returned as
model inputs, used as training targets, or used to construct a correction curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from disentangle_attempt.dataset import CrossSectorPatch, grid_row


HERE = Path(__file__).resolve().parent
DEFAULT_BASE_PARQUET = HERE / "data" / "cross_sector_raw.parquet"
DEFAULT_REFERENCE_SPLIT = (
    HERE / "outputs" / "local_s1_c4_ccd2_12px_p128_i32" / "split_tics.json"
)

SPLITS = ("train", "val", "test")
TARGET_CHIP = (1, 4, 2)
SPREAD_BANDS = ((128.0, 384.0), (384.0, 768.0))
SPACING_LADDER = (256.0, 192.0, 128.0)

# These are the complete, uncapped peer-pool TIC sets reconstructed by the baseline
# seed-42 split.  The hashes make accidental repartitioning fail loudly.
EXPECTED_LEGACY_POOL_COUNTS = {"train": 1590, "val": 199, "test": 199}
EXPECTED_LEGACY_POOL_SHA256 = {
    "train": "e884b7cd2aaae176acc42c9c5ba7a09a6458edc76a7e4ad300a94a9765a6acc4",
    "val": "5601cceabfae2d1cc7f207711058743ef06907b5e367808ae46686367828e9c0",
    "test": "f65bd7e9ef47a7fed3380b9c1c4c4ab2335c31048451eb298ef4f12bf8fabee3",
}
EXPECTED_ANCHOR_COUNTS = {"train": 320, "val": 40, "test": 40}


def canonical_numeric_tic(value: Any) -> str:
    """Return a canonical unsigned decimal TIC string or raise ``ValueError``."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError("missing TIC")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"non-numeric TIC {value!r}")
    number = int(text)
    if number <= 0:
        raise ValueError(f"non-positive TIC {value!r}")
    return str(number)


def tic_set_sha256(tics: Iterable[Any]) -> str:
    """Hash a TIC set in numeric order using the frozen manifest convention."""
    values = sorted({canonical_numeric_tic(value) for value in tics}, key=int)
    payload = "".join(f"{value}\n" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def stable_peer_split(tic: Any, seed: int = 42) -> dict[str, Any]:
    """Stable 80/10/10 assignment used *only* for newly acquired peer TICs.

    This is intentionally a per-key hash rather than a permutation of the complete
    catalog.  A later catalog append cannot change any assignment already made.
    """
    canonical = canonical_numeric_tic(tic)
    hash_key = f"{int(seed)}:{canonical}"
    digest = hashlib.sha256(hash_key.encode("ascii")).digest()
    value_uint64 = int.from_bytes(digest[:8], "big", signed=False)
    value = value_uint64 / float(1 << 64)
    split = "train" if value < 0.8 else ("val" if value < 0.9 else "test")
    return {
        "TIC": canonical,
        "split": split,
        "hash_key": hash_key,
        "hash_sha256": digest.hex(),
        "hash_uint64": value_uint64,
        "hash_value_unit_interval": value,
        "hash_scheme": "sha256-first-u64-big-endian-v1",
    }


def _safe_gaia(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _circular_max_gap(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 2.0 * np.pi
    ordered = np.sort(np.mod(np.asarray(angles, dtype=float), 2.0 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    return float(gaps.max())


@dataclass(frozen=True)
class _PreparedPeer:
    source_row: int
    tic: str
    gaia: int | None
    detector_x: float
    detector_y: float
    curve: np.ndarray
    valid: np.ndarray
    flagged: np.ndarray
    background: np.ndarray
    tess_flags: np.ndarray
    tglc_flags: np.ndarray

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


class ExpandedSpreadPeerPatch(CrossSectorPatch):
    """Exact legacy anchors plus disjoint supplemental peer-only pools.

    Infeasible anchors are recorded but never removed from ``split_anchors``.  Call
    :meth:`assert_training_ready` before constructing training datasets; this enforces
    the latest requirement that the exact 320/40/40 anchor split remain unchanged.
    """

    def __init__(
        self,
        base_parquet_path: str | Path,
        supplemental_parquet_path: str | Path,
        *,
        reference_split_path: str | Path = DEFAULT_REFERENCE_SPLIT,
        curve_length: int = 1024,
        n_peers: int = 8,
        min_valid_fraction: float = 0.5,
        split_seed: int = 42,
        max_eligible_anchors: int = 400,
        min_chip_anchors: int = 64,
        bands: Iterable[Iterable[float]] = SPREAD_BANDS,
        peers_per_band: int = 4,
        spacing_ladder: Iterable[float] = SPACING_LADDER,
        outer_expansion_radii: Iterable[float] = (),
        exact_search_node_budget: int = 1_000_000,
        strict_legacy_manifest: bool = True,
        verbose: bool = True,
    ):
        if int(n_peers) != 8:
            raise ValueError("spread experiment requires exactly eight peers")
        self.spread_bands = tuple(tuple(float(x) for x in band) for band in bands)
        self.peers_per_band = int(peers_per_band)
        self.spacing_ladder = tuple(float(x) for x in spacing_ladder)
        self.outer_expansion_radii = tuple(float(x) for x in outer_expansion_radii)
        self.exact_search_node_budget = int(exact_search_node_budget)
        if self.spread_bands != SPREAD_BANDS:
            raise ValueError(f"distance bands must remain literal {SPREAD_BANDS}")
        if self.peers_per_band != 4:
            raise ValueError("spread experiment requires four peers in each band")
        if self.spacing_ladder != SPACING_LADDER:
            raise ValueError(f"spacing ladder must remain literal {SPACING_LADDER}")
        previous_cap = self.spread_bands[-1][1]
        for cap in self.outer_expansion_radii:
            if cap <= previous_cap:
                raise ValueError("outer expansion radii must increase beyond 768 pixels")
            previous_cap = cap

        self.base_parquet_path = str(Path(base_parquet_path).expanduser().resolve())
        self.supplemental_parquet_path = str(
            Path(supplemental_parquet_path).expanduser().resolve()
        )
        self.reference_split_path = str(
            Path(reference_split_path).expanduser().resolve()
        )
        self.split_seed = int(split_seed)

        # Rebuild the baseline *before* supplemental rows are visible.  In particular,
        # this freezes the cadence grid, legacy 80/10/10 TIC assignment, anchor rows,
        # and cap selection.
        super().__init__(
            self.base_parquet_path,
            target_sector=TARGET_CHIP[0],
            camera=TARGET_CHIP[1],
            ccd=TARGET_CHIP[2],
            curve_length=int(curve_length),
            n_peers=int(n_peers),
            min_valid_fraction=float(min_valid_fraction),
            split_seed=self.split_seed,
            max_eligible_anchors=int(max_eligible_anchors),
            min_chip_anchors=int(min_chip_anchors),
            require_cross_sector=False,
            peer_min_distance=12.0,
            verbose=False,
        )
        if tuple(self.target) != TARGET_CHIP:
            raise AssertionError(f"unexpected target chip {self.target}")

        self._validate_fixed_legacy(reference_split_path, strict_legacy_manifest)
        self.legacy_nearest_peers = {
            split: (rows.copy(), distances.copy())
            for split, (rows, distances) in self.peers.items()
        }
        self._ingest_supplemental(min_valid_fraction=float(min_valid_fraction))
        self._build_combined_split_pools()
        self._build_spread_peer_tables()
        self._assert_partition_contract()
        self._assert_selected_geometry()

        if verbose:
            self.report_spread_feasibility()

    # ----------------------------------------------------------- frozen baseline
    def _validate_fixed_legacy(
        self, reference_split_path: str | Path, strict_manifest: bool
    ) -> None:
        with Path(reference_split_path).expanduser().open() as handle:
            reference = json.load(handle)

        self.fixed_anchor_tics: dict[str, tuple[str, ...]] = {}
        self.legacy_split_tics: dict[str, set[str]] = {}
        self.legacy_split_pool: dict[str, dict[tuple[int, int, int], np.ndarray]] = {}
        for split in SPLITS:
            expected = tuple(canonical_numeric_tic(value) for value in reference[split])
            actual = tuple(canonical_numeric_tic(value)
                           for value in self.tic[self.split_anchors[split]])
            if len(actual) != EXPECTED_ANCHOR_COUNTS[split] or set(actual) != set(expected):
                raise AssertionError(
                    f"{split} anchors drifted: {len(actual)} actual; reference has "
                    f"{len(expected)}"
                )
            self.fixed_anchor_tics[split] = actual
            legacy = {canonical_numeric_tic(value) for value in self.split_tics[split]}
            self.legacy_split_tics[split] = legacy
            self.legacy_split_pool[split] = {
                chip: rows.copy() for chip, rows in self.split_pool[split].items()
            }
            if strict_manifest:
                if len(legacy) != EXPECTED_LEGACY_POOL_COUNTS[split]:
                    raise AssertionError(f"{split} legacy peer-pool count drifted")
                if tic_set_sha256(legacy) != EXPECTED_LEGACY_POOL_SHA256[split]:
                    raise AssertionError(f"{split} legacy peer-pool TIC hash drifted")

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if self.legacy_split_tics[left] & self.legacy_split_tics[right]:
                raise AssertionError(f"legacy {left}/{right} TIC leakage")
        self.legacy_all_tics = set().union(*self.legacy_split_tics.values())
        self.fixed_anchor_all_tics = set().union(
            *(set(values) for values in self.fixed_anchor_tics.values())
        )
        if not self.fixed_anchor_all_tics <= self.legacy_all_tics:
            raise AssertionError("a fixed anchor is absent from the frozen legacy pools")

    # ------------------------------------------------------ supplemental ingestion
    def _legacy_gaia_ids(self) -> set[int]:
        columns = ["TIC", "GAIADR3", "sector", "camera", "ccd"]
        frame = pd.read_parquet(self.base_parquet_path, columns=columns)
        frame["TIC"] = frame["TIC"].map(canonical_numeric_tic)
        frame = frame.drop_duplicates(["TIC", "sector"])
        frame = frame[
            (frame["sector"].astype(int) == TARGET_CHIP[0])
            & (frame["camera"].astype(int) == TARGET_CHIP[1])
            & (frame["ccd"].astype(int) == TARGET_CHIP[2])
            & frame["TIC"].isin(self.legacy_all_tics)
        ]
        return {value for value in (_safe_gaia(x) for x in frame["GAIADR3"]) if value}

    def _ingest_supplemental(self, min_valid_fraction: float) -> None:
        path = Path(self.supplemental_parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"supplemental peer parquet does not exist: {path}")
        frame = pd.read_parquet(path).reset_index(drop=True)
        required = {
            "TIC", "sector", "camera", "ccd", "DETECTOR_X", "DETECTOR_Y",
            "cadence_num", "flux",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"supplemental parquet missing columns {sorted(missing)}")

        audit: list[dict[str, Any]] = []
        prepared: list[_PreparedPeer] = []
        minimum_valid = int(float(min_valid_fraction) * self.curve_length)
        legacy_gaia = self._legacy_gaia_ids() if "GAIADR3" in frame.columns else set()

        for source_row, row in frame.iterrows():
            record: dict[str, Any] = {
                "source_row": int(source_row),
                "TIC": None,
                "GAIADR3": _safe_gaia(row.get("GAIADR3")),
                "sector": row.get("sector"),
                "camera": row.get("camera"),
                "ccd": row.get("ccd"),
                "status": "pending",
                "reason": "",
                "declared_peer_pool": row.get("peer_pool"),
            }
            try:
                tic = canonical_numeric_tic(row.get("TIC"))
                record["TIC"] = tic
            except ValueError as exc:
                record.update(status="excluded", reason=f"invalid_tic:{exc}")
                audit.append(record)
                continue
            try:
                chip = (int(row["sector"]), int(row["camera"]), int(row["ccd"]))
            except (TypeError, ValueError, OverflowError):
                record.update(status="excluded", reason="invalid_chip_metadata")
                audit.append(record)
                continue
            if chip != TARGET_CHIP:
                record.update(status="excluded", reason="outside_sector1_camera4_ccd2")
                audit.append(record)
                continue
            if tic in self.legacy_all_tics:
                record.update(status="excluded", reason="legacy_tic_uses_frozen_row")
                audit.append(record)
                continue
            declared_pool = row.get("peer_pool")
            if declared_pool is not None and not pd.isna(declared_pool):
                expected_pool = stable_peer_split(tic, self.split_seed)["split"]
                if str(declared_pool) != expected_pool:
                    raise AssertionError(
                        f"supplemental TIC {tic} declares pool {declared_pool!r}, "
                        f"stable hash requires {expected_pool!r}"
                    )
            gaia = record["GAIADR3"]
            if gaia is not None and gaia in legacy_gaia:
                record.update(status="excluded", reason="legacy_gaia_alias")
                audit.append(record)
                continue
            try:
                detector_x = float(row["DETECTOR_X"])
                detector_y = float(row["DETECTOR_Y"])
            except (TypeError, ValueError, OverflowError):
                detector_x = detector_y = float("nan")
            if not (np.isfinite(detector_x) and np.isfinite(detector_y)):
                record.update(status="excluded", reason="nonfinite_detector_coordinate")
                audit.append(record)
                continue
            try:
                curve, valid, flagged, background, tess_flags, tglc_flags = grid_row(
                    row, self.grids[TARGET_CHIP[0]], self.curve_length
                )
            except Exception as exc:  # keep the acquisition audit actionable
                record.update(status="excluded", reason=f"preprocess_error:{type(exc).__name__}")
                audit.append(record)
                continue
            record["n_valid"] = int(valid.sum())
            if int(valid.sum()) < minimum_valid:
                record.update(status="excluded", reason="below_min_valid_fraction")
                audit.append(record)
                continue
            record.update(status="candidate", reason="")
            audit.append(record)
            prepared.append(_PreparedPeer(
                source_row=int(source_row), tic=tic, gaia=gaia,
                detector_x=detector_x, detector_y=detector_y,
                curve=curve, valid=valid, flagged=flagged, background=background,
                tess_flags=tess_flags, tglc_flags=tglc_flags,
            ))

        audit_by_source = {int(item["source_row"]): item for item in audit}

        # One deterministic curve per TIC.  Valid coverage is a normal data-quality
        # criterion, not a morphology/correlation selector.
        unique_tic: list[_PreparedPeer] = []
        by_tic: dict[str, list[_PreparedPeer]] = {}
        for item in prepared:
            by_tic.setdefault(item.tic, []).append(item)
        for tic in sorted(by_tic, key=int):
            group = sorted(
                by_tic[tic],
                key=lambda x: (
                    -x.n_valid,
                    x.gaia if x.gaia is not None else (1 << 63),
                    x.detector_x,
                    x.detector_y,
                    x.source_row,
                ),
            )
            unique_tic.append(group[0])
            audit_by_source[group[0].source_row].update(
                status="selected", reason="canonical_row_for_tic"
            )
            for duplicate in group[1:]:
                audit_by_source[duplicate.source_row].update(
                    status="excluded", reason="duplicate_tic_noncanonical_row"
                )

        # A Gaia alias is the same source even if malformed metadata gives it multiple
        # TIC labels.  Keep one canonical TIC so it cannot cross pools or occupy two
        # peer slots.
        by_gaia: dict[int, list[_PreparedPeer]] = {}
        for item in unique_tic:
            if item.gaia is not None:
                by_gaia.setdefault(item.gaia, []).append(item)
        dropped_alias_sources: set[int] = set()
        for gaia, group in by_gaia.items():
            if len(group) < 2:
                continue
            keep = min(group, key=lambda item: (int(item.tic), item.source_row))
            for alias in group:
                if alias is keep:
                    continue
                dropped_alias_sources.add(alias.source_row)
                audit_by_source[alias.source_row].update(
                    status="excluded", reason="duplicate_gaia_alias_noncanonical_tic"
                )
        selected = [item for item in unique_tic if item.source_row not in dropped_alias_sources]
        selected.sort(key=lambda item: int(item.tic))

        self.supplemental_ingestion_audit = pd.DataFrame(audit).sort_values("source_row")
        self.expanded_split_tics = {split: set() for split in SPLITS}
        self.expanded_partition_records: list[dict[str, Any]] = []
        self.new_row_by_tic: dict[str, int] = {}
        self.peer_pool_source_by_row = {
            int(row): "legacy" for split in SPLITS
            for row in self.legacy_split_pool[split][TARGET_CHIP]
        }

        if not selected:
            self._selected_supplemental = []
            self.peer_pool_assignment = self._pool_assignment_frame()
            return

        start = len(self.tic)
        self.X = np.concatenate([self.X, np.stack([item.curve for item in selected])])
        self.M = np.concatenate([self.M, np.stack([item.valid for item in selected])])
        self.Q = np.concatenate([self.Q, np.stack([item.flagged for item in selected])])
        self.BG = np.concatenate([self.BG, np.stack([item.background for item in selected])])
        self.F = np.concatenate([self.F, np.stack([item.tess_flags for item in selected])])
        self.G = np.concatenate([self.G, np.stack([item.tglc_flags for item in selected])])
        self.tic = np.concatenate([self.tic, np.asarray([item.tic for item in selected])])
        self.tic_int = np.concatenate([
            self.tic_int, np.asarray([int(item.tic) for item in selected], dtype=np.int64)
        ])
        self.sector = np.concatenate([
            self.sector, np.full(len(selected), TARGET_CHIP[0], dtype=self.sector.dtype)
        ])
        self.camera = np.concatenate([
            self.camera, np.full(len(selected), TARGET_CHIP[1], dtype=self.camera.dtype)
        ])
        self.ccd = np.concatenate([
            self.ccd, np.full(len(selected), TARGET_CHIP[2], dtype=self.ccd.dtype)
        ])
        self.det_x = np.concatenate([
            self.det_x, np.asarray([item.detector_x for item in selected], dtype=float)
        ])
        self.det_y = np.concatenate([
            self.det_y, np.asarray([item.detector_y for item in selected], dtype=float)
        ])
        self.n_valid = np.concatenate([
            self.n_valid, np.asarray([item.n_valid for item in selected], dtype=np.int64)
        ])

        for offset, item in enumerate(selected):
            row_index = start + offset
            assignment = stable_peer_split(item.tic, self.split_seed)
            assignment.update({
                "row": row_index,
                "GAIADR3": item.gaia,
                "provenance": "expanded_hash_v1",
                "source_row": item.source_row,
            })
            self.expanded_partition_records.append(assignment)
            self.expanded_split_tics[assignment["split"]].add(item.tic)
            self.new_row_by_tic[item.tic] = row_index
            self.peer_pool_source_by_row[row_index] = "expanded"
            self.rows_by_tic.setdefault(item.tic, []).append(row_index)
            self.chip_of_row[row_index] = TARGET_CHIP
        self._selected_supplemental = selected
        self.peer_pool_assignment = self._pool_assignment_frame()

    def _pool_assignment_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for split in SPLITS:
            for tic in sorted(self.legacy_split_tics[split], key=int):
                legacy_rows = self.legacy_split_pool[split][TARGET_CHIP]
                matching = legacy_rows[self.tic[legacy_rows].astype(str) == tic]
                rows.append({
                    "TIC": tic, "split": split, "row": int(matching[0]),
                    "provenance": "legacy_frozen", "hash_key": "",
                    "hash_sha256": "", "hash_uint64": None,
                    "hash_value_unit_interval": None,
                    "hash_scheme": "legacy_numpy_rng_seed42_frozen",
                    "GAIADR3": None, "source_row": None,
                })
        rows.extend(self.expanded_partition_records)
        return pd.DataFrame(rows).sort_values(["split", "TIC"], kind="stable")

    def _build_combined_split_pools(self) -> None:
        self.split_tics = {}
        self.split_pool = {}
        for split in SPLITS:
            new_tics = self.expanded_split_tics[split]
            new_rows = np.asarray(
                [self.new_row_by_tic[tic] for tic in sorted(new_tics, key=int)],
                dtype=np.int64,
            )
            legacy_rows = self.legacy_split_pool[split][TARGET_CHIP]
            self.split_tics[split] = set(self.legacy_split_tics[split]) | set(new_tics)
            self.split_pool[split] = {
                TARGET_CHIP: np.concatenate([legacy_rows, new_rows]).astype(np.int64)
            }

    # --------------------------------------------------------- geometry selection
    def _candidate_band(
        self, anchor: int, pool: np.ndarray, band_index: int, outer_cap: float = 768.0
    ) -> np.ndarray:
        low, high = self.spread_bands[band_index]
        if band_index == 1:
            high = float(outer_cap)
        rows = np.asarray(pool, dtype=np.int64)
        dx = self.det_x[rows] - self.det_x[anchor]
        dy = self.det_y[rows] - self.det_y[anchor]
        distance = np.hypot(dx, dy)
        different = self.tic[rows].astype(str) != str(self.tic[anchor])
        finite = np.isfinite(distance) & np.isfinite(dx) & np.isfinite(dy)
        inside = (distance >= low) & (distance < high if band_index == 0 else distance <= high)
        rows = rows[different & finite & inside]
        if not len(rows):
            return rows
        dx = self.det_x[rows] - self.det_x[anchor]
        dy = self.det_y[rows] - self.det_y[anchor]
        distance = np.hypot(dx, dy)
        angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        order = np.lexsort((self.tic_int[rows], distance, angle))
        return rows[order]

    def _compatible(self, rows: np.ndarray, selected: list[int], spacing: float) -> np.ndarray:
        if not selected or not len(rows):
            return np.ones(len(rows), dtype=bool)
        xy = np.column_stack([self.det_x[rows], self.det_y[rows]])
        chosen_xy = np.column_stack([
            self.det_x[np.asarray(selected, dtype=np.int64)],
            self.det_y[np.asarray(selected, dtype=np.int64)],
        ])
        return (np.linalg.norm(xy[:, None, :] - chosen_xy[None, :, :], axis=2)
                >= spacing - 1e-9).all(axis=1)

    def _seed_rows(self, candidates: tuple[np.ndarray, np.ndarray]) -> list[int]:
        seeds: list[int] = []
        for rows in candidates:
            if not len(rows):
                continue
            count = min(24, len(rows))
            indices = np.unique(np.linspace(0, len(rows) - 1, count).round().astype(int))
            seeds.extend(int(rows[index]) for index in indices)
        return list(dict.fromkeys(seeds))

    def _greedy_from_seed(
        self,
        anchor: int,
        candidates: tuple[np.ndarray, np.ndarray],
        spacing: float,
        seed_row: int,
    ) -> list[int] | None:
        band_of = {
            int(row): band for band, rows in enumerate(candidates) for row in rows
        }
        selected = [int(seed_row)]
        counts = [0, 0]
        counts[band_of[int(seed_row)]] = 1
        schedule = [0, 1] * self.peers_per_band
        while len(selected) < self.n_peers:
            band = next((value for value in schedule if counts[value] < self.peers_per_band), None)
            if band is None:
                break
            rows = candidates[band]
            allowed = self._compatible(rows, selected, spacing)
            if selected:
                allowed &= ~np.isin(rows, np.asarray(selected, dtype=np.int64))
            available = rows[allowed]
            if not len(available):
                # Let the other band go next if it still has quota; this avoids one
                # fixed alternation becoming the reason a valid group is missed.
                other = 1 - band
                if counts[other] >= self.peers_per_band:
                    return None
                available = candidates[other][self._compatible(
                    candidates[other], selected, spacing
                )]
                available = available[~np.isin(available, np.asarray(selected))]
                if not len(available):
                    return None
                band = other
            chosen_xy = np.column_stack([
                self.det_x[np.asarray(selected)], self.det_y[np.asarray(selected)]
            ])
            xy = np.column_stack([self.det_x[available], self.det_y[available]])
            min_space = np.linalg.norm(xy[:, None] - chosen_xy[None], axis=2).min(axis=1)
            angle = np.mod(np.arctan2(
                self.det_y[available] - self.det_y[anchor],
                self.det_x[available] - self.det_x[anchor],
            ), 2.0 * np.pi)
            selected_angle = np.mod(np.arctan2(
                self.det_y[np.asarray(selected)] - self.det_y[anchor],
                self.det_x[np.asarray(selected)] - self.det_x[anchor],
            ), 2.0 * np.pi)
            delta = np.abs(angle[:, None] - selected_angle[None])
            min_angle = np.minimum(delta, 2.0 * np.pi - delta).min(axis=1)
            anchor_distance = np.hypot(
                self.det_x[available] - self.det_x[anchor],
                self.det_y[available] - self.det_y[anchor],
            )
            # np.lexsort uses the final key as primary.  Small TIC is the final,
            # deterministic tie breaker after spatial and angular coverage.
            order = np.lexsort((
                self.tic_int[available], -anchor_distance, -min_angle, -min_space
            ))
            selected.append(int(available[order[0]]))
            counts[band] += 1
        return selected if counts == [self.peers_per_band, self.peers_per_band] else None

    def _solution_score(self, anchor: int, rows: list[int]) -> tuple[Any, ...]:
        selected = np.asarray(rows, dtype=np.int64)
        xy = np.column_stack([self.det_x[selected], self.det_y[selected]])
        pair = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        triangle = pair[np.triu_indices(len(selected), 1)]
        angles = np.mod(np.arctan2(
            self.det_y[selected] - self.det_y[anchor],
            self.det_x[selected] - self.det_x[anchor],
        ), 2.0 * np.pi)
        # Angular coverage is primary; pairwise spatial spread breaks ties.
        return (
            -_circular_max_gap(angles),
            -_circular_max_gap(angles[:self.peers_per_band]),
            -_circular_max_gap(angles[self.peers_per_band:]),
            float(triangle.min()),
            float(triangle.sum()),
            tuple(-int(self.tic_int[row]) for row in selected),
        )

    def _greedy_fps(
        self, anchor: int, candidates: tuple[np.ndarray, np.ndarray], spacing: float
    ) -> list[int] | None:
        solutions: list[list[int]] = []
        for seed in self._seed_rows(candidates):
            selected = self._greedy_from_seed(anchor, candidates, spacing, seed)
            if selected is None:
                continue
            # Public order is band then angle, irrespective of greedy traversal.
            ordered: list[int] = []
            for band_rows in candidates:
                chosen = [row for row in selected if row in set(map(int, band_rows))]
                chosen.sort(key=lambda row: (
                    float(np.mod(np.arctan2(
                        self.det_y[row] - self.det_y[anchor],
                        self.det_x[row] - self.det_x[anchor],
                    ), 2.0 * np.pi)),
                    float(np.hypot(self.det_x[row] - self.det_x[anchor],
                                   self.det_y[row] - self.det_y[anchor])),
                    int(self.tic_int[row]),
                ))
                ordered.extend(chosen)
            if len(ordered) == self.n_peers:
                solutions.append(ordered)
        return max(solutions, key=lambda rows: self._solution_score(anchor, rows)) \
            if solutions else None

    def _exact_feasible_search(
        self, candidates: tuple[np.ndarray, np.ndarray], spacing: float
    ) -> tuple[list[int] | None, int, bool]:
        """Bounded deterministic backtracking used only if multi-start FPS misses.

        ``exhausted`` means the node budget was hit, not that geometry was infeasible;
        callers must never relax spacing after that ambiguous outcome.
        """
        inner, outer = candidates
        nodes = 0
        budget_hit = False

        def choose_outer(pool: np.ndarray, start: int, picked: list[int]) -> list[int] | None:
            nonlocal nodes, budget_hit
            remaining = self.peers_per_band - len(picked)
            if remaining == 0:
                return list(picked)
            if len(pool) - start < remaining:
                return None
            for position in range(start, len(pool) - remaining + 1):
                nodes += 1
                if nodes > self.exact_search_node_budget:
                    budget_hit = True
                    return None
                row = int(pool[position])
                if not self._compatible(np.asarray([row]), picked, spacing)[0]:
                    continue
                found = choose_outer(pool, position + 1, picked + [row])
                if found is not None:
                    return found
                if budget_hit:
                    return None
            return None

        def choose_inner(
            start: int, picked: list[int], compatible_outer: np.ndarray
        ) -> list[int] | None:
            nonlocal nodes, budget_hit
            remaining = self.peers_per_band - len(picked)
            if remaining == 0:
                outer_solution = choose_outer(compatible_outer, 0, [])
                return list(picked) + outer_solution if outer_solution is not None else None
            if len(inner) - start < remaining or len(compatible_outer) < self.peers_per_band:
                return None
            for position in range(start, len(inner) - remaining + 1):
                nodes += 1
                if nodes > self.exact_search_node_budget:
                    budget_hit = True
                    return None
                row = int(inner[position])
                if not self._compatible(np.asarray([row]), picked, spacing)[0]:
                    continue
                next_outer = compatible_outer[
                    self._compatible(compatible_outer, [row], spacing)
                ]
                if len(next_outer) < self.peers_per_band:
                    continue
                found = choose_inner(position + 1, picked + [row], next_outer)
                if found is not None:
                    return found
                if budget_hit:
                    return None
            return None

        result = choose_inner(0, [], outer)
        return result, nodes, budget_hit

    def _select_spread_for_anchor(
        self, anchor: int, pool: np.ndarray
    ) -> tuple[list[int] | None, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        caps = (self.spread_bands[-1][1],) + self.outer_expansion_radii
        stages = [
            (self.spread_bands[-1][1], spacing, level)
            for level, spacing in enumerate(self.spacing_ladder)
        ]
        # Expansion is permitted only after the complete 256 -> 192 -> 128 ladder at
        # the literal 768-pixel cap.  Keep the already-relaxed 128-pixel spacing while
        # increasing only the outer upper boundary.
        stages.extend(
            (cap, self.spacing_ladder[-1], len(self.spacing_ladder) + index)
            for index, cap in enumerate(caps[1:])
        )
        inner_candidates = self._candidate_band(anchor, pool, 0)
        for outer_cap, spacing, level in stages:
            candidates = (
                inner_candidates,
                self._candidate_band(anchor, pool, 1, outer_cap=outer_cap),
            )
            attempt = {
                "anchor_row": int(anchor),
                "anchor_TIC": str(self.tic[anchor]),
                "spacing_tier": spacing,
                "spacing_fallback_level": level,
                "inner_candidate_count": int(len(candidates[0])),
                "outer_candidate_count": int(len(candidates[1])),
                "outer_radius_cap": float(outer_cap),
                "expansion_used": bool(outer_cap > self.spread_bands[-1][1]),
                "search_nodes": 0,
                "status": "pending",
                "relaxation_reason": "",
            }
            if min(len(candidates[0]), len(candidates[1])) < self.peers_per_band:
                attempt["status"] = "infeasible_candidate_count"
                attempts.append(attempt)
                continue
            solution = self._greedy_fps(anchor, candidates, spacing)
            if solution is not None:
                attempt["status"] = "selected_farthest_point"
                attempts.append(attempt)
                return solution, attempts
            solution, nodes, budget_hit = self._exact_feasible_search(candidates, spacing)
            attempt["search_nodes"] = nodes
            if budget_hit:
                attempt["status"] = "unresolved_search_budget"
                attempt["relaxation_reason"] = (
                    "deterministic exact-search node budget exhausted; fallback recorded"
                )
                attempts.append(attempt)
                # This is not labelled mathematically infeasible.  The next permitted
                # tier is nevertheless attempted so one difficult anchor cannot block
                # a fully explicit pre-training feasibility audit.
                continue
            if solution is not None:
                attempt["status"] = "selected_exact_fallback"
                attempts.append(attempt)
                ordered: list[int] = []
                for band_index, band_rows in enumerate(candidates):
                    band_set = set(map(int, band_rows))
                    chosen = [row for row in solution if row in band_set]
                    chosen.sort(key=lambda row: (
                        float(np.mod(np.arctan2(
                            self.det_y[row] - self.det_y[anchor],
                            self.det_x[row] - self.det_x[anchor],
                        ), 2.0 * np.pi)),
                        float(np.hypot(self.det_x[row] - self.det_x[anchor],
                                       self.det_y[row] - self.det_y[anchor])),
                        int(self.tic_int[row]),
                    ))
                    ordered.extend(chosen)
                return ordered, attempts
            attempt["status"] = "infeasible_exhaustive_search"
            attempt["relaxation_reason"] = "no feasible unique-eight group at this stage"
            attempts.append(attempt)
        return None, attempts

    def _build_spread_peer_tables(self) -> None:
        selected_records: list[dict[str, Any]] = []
        attempt_records: list[dict[str, Any]] = []
        self.excluded_anchors: dict[str, list[int]] = {split: [] for split in SPLITS}
        self.selection_records_by_anchor: dict[int, list[dict[str, Any]]] = {}
        spread_tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for split in SPLITS:
            anchors = self.split_anchors[split]
            rows_out = np.full((len(anchors), self.n_peers), -1, dtype=np.int64)
            distances_out = np.full((len(anchors), self.n_peers), np.nan, dtype=np.float32)
            pool = self.split_pool[split][TARGET_CHIP]
            for anchor_index, anchor_value in enumerate(anchors):
                anchor = int(anchor_value)
                selected, attempts = self._select_spread_for_anchor(anchor, pool)
                for attempt in attempts:
                    attempt["split"] = split
                attempt_records.extend(attempts)
                if selected is None:
                    self.excluded_anchors[split].append(anchor)
                    continue
                spacing = float(attempts[-1]["spacing_tier"])
                selected_cap = float(attempts[-1]["outer_radius_cap"])
                expansion_used = bool(attempts[-1]["expansion_used"])
                selected_array = np.asarray(selected, dtype=np.int64)
                distance = np.hypot(
                    self.det_x[selected_array] - self.det_x[anchor],
                    self.det_y[selected_array] - self.det_y[anchor],
                )
                rows_out[anchor_index] = selected_array
                distances_out[anchor_index] = distance.astype(np.float32)
                xy = np.column_stack([
                    self.det_x[selected_array], self.det_y[selected_array]
                ])
                pair = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
                group_minimum = float(pair[np.triu_indices(self.n_peers, 1)].min())
                records: list[dict[str, Any]] = []
                for peer_index, (peer, anchor_distance) in enumerate(
                    zip(selected_array, distance), start=1
                ):
                    dx = float(self.det_x[peer] - self.det_x[anchor])
                    dy = float(self.det_y[peer] - self.det_y[anchor])
                    angle = float(np.mod(np.arctan2(dy, dx), 2.0 * np.pi))
                    band_index = 0 if peer_index <= self.peers_per_band else 1
                    low, high = self.spread_bands[band_index]
                    if band_index == 1:
                        high = selected_cap
                    record = {
                        "split": split,
                        "anchor_row": anchor,
                        "anchor_TIC": str(self.tic[anchor]),
                        "anchor_x": float(self.det_x[anchor]),
                        "anchor_y": float(self.det_y[anchor]),
                        "peer_order": peer_index,
                        "peer_row": int(peer),
                        "peer_TIC": str(self.tic[peer]),
                        "peer_x": float(self.det_x[peer]),
                        "peer_y": float(self.det_y[peer]),
                        "peer_pool_provenance": self.peer_pool_source_by_row[int(peer)],
                        "band_index": band_index,
                        "radial_band": f"[{low:g},{high:g}{')' if band_index == 0 else ']'}",
                        "anchor_distance": float(anchor_distance),
                        "detector_angle_radians": angle,
                        "detector_angle_degrees": float(np.degrees(angle)),
                        "spacing_tier": spacing,
                        "spacing_fallback_level": int(attempts[-1]["spacing_fallback_level"]),
                        "fallback_stage": (
                            f"spacing_{int(spacing)}_cap_{int(selected_cap)}"
                        ),
                        "outer_radius_cap": selected_cap,
                        "expansion_used": expansion_used,
                        "group_minimum_peer_separation": group_minimum,
                        "selection_method": attempts[-1]["status"],
                    }
                    records.append(record)
                    selected_records.append(record)
                self.selection_records_by_anchor[anchor] = records
            spread_tables[split] = rows_out, distances_out

        self.peers = spread_tables
        self.peer_selection_audit = pd.DataFrame(selected_records)
        self.peer_selection_attempts = pd.DataFrame(attempt_records)
        exclusion_rows = []
        for split in SPLITS:
            for anchor in self.excluded_anchors[split]:
                anchor_attempts = [record for record in attempt_records
                                   if record["split"] == split
                                   and int(record["anchor_row"]) == anchor]
                exclusion_rows.append({
                    "split": split,
                    "anchor_row": anchor,
                    "anchor_TIC": str(self.tic[anchor]),
                    "reason": anchor_attempts[-1]["status"] if anchor_attempts else "unknown",
                    "attempt_count": len(anchor_attempts),
                })
        self.excluded_anchor_records = pd.DataFrame(exclusion_rows)

    # --------------------------------------------------------------- assertions
    def _assert_partition_contract(self) -> None:
        anchor_union = set().union(
            *(set(values) for values in self.fixed_anchor_tics.values())
        )
        new_union = set().union(*self.expanded_split_tics.values())
        if new_union & self.legacy_all_tics:
            raise AssertionError("a supplemental TIC overlaps a frozen legacy TIC")
        if new_union & anchor_union:
            raise AssertionError("a supplemental peer-only TIC overlaps an anchor TIC")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if self.split_tics[left] & self.split_tics[right]:
                raise AssertionError(f"combined {left}/{right} TIC leakage")
        if len(new_union) != len(self.new_row_by_tic):
            raise AssertionError("new TIC does not map one-to-one to a peer row")
        for record in self.expanded_partition_records:
            recomputed = stable_peer_split(record["TIC"], self.split_seed)
            if recomputed["split"] != record["split"]:
                raise AssertionError("supplemental hash assignment is not reproducible")
        for split in SPLITS:
            if len(self.split_anchors[split]) != EXPECTED_ANCHOR_COUNTS[split]:
                raise AssertionError("fixed anchor count changed after peer expansion")
            actual_anchor = {str(value) for value in self.tic[self.split_anchors[split]]}
            if actual_anchor != set(self.fixed_anchor_tics[split]):
                raise AssertionError("fixed anchor identity changed after peer expansion")
            pool = self.split_pool[split][TARGET_CHIP]
            pool_tics = {str(value) for value in self.tic[pool]}
            if pool_tics != self.split_tics[split]:
                raise AssertionError(f"{split} row pool and TIC manifest disagree")

    def _assert_selected_geometry(self) -> None:
        if self.peer_selection_audit.empty:
            return
        for (split, anchor_tic), group in self.peer_selection_audit.groupby(
            ["split", "anchor_TIC"], sort=False
        ):
            group = group.sort_values("peer_order")
            if len(group) != self.n_peers or group["peer_TIC"].nunique() != self.n_peers:
                raise AssertionError(f"{split} TIC {anchor_tic}: peer group is not unique-eight")
            if str(anchor_tic) in set(group["peer_TIC"].astype(str)):
                raise AssertionError(f"{split} TIC {anchor_tic}: anchor selected as peer")
            inner = group.iloc[:self.peers_per_band]["anchor_distance"].to_numpy(float)
            outer = group.iloc[self.peers_per_band:]["anchor_distance"].to_numpy(float)
            if not ((inner >= 128.0) & (inner < 384.0)).all():
                raise AssertionError(f"{split} TIC {anchor_tic}: inner band narrowed/drifted")
            cap = float(group["outer_radius_cap"].iloc[0])
            expanded = bool(group["expansion_used"].iloc[0])
            if cap < 768.0 or (cap > 768.0) != expanded:
                raise AssertionError(f"{split} TIC {anchor_tic}: expansion audit is inconsistent")
            if not ((outer >= 384.0) & (outer <= cap)).all():
                raise AssertionError(f"{split} TIC {anchor_tic}: outer band/cap violated")
            tier = float(group["spacing_tier"].iloc[0])
            if tier not in self.spacing_ladder:
                raise AssertionError("unapproved spacing tier")
            if float(group["group_minimum_peer_separation"].iloc[0]) < tier - 1e-6:
                raise AssertionError(f"{split} TIC {anchor_tic}: spacing tier violated")
            allowed = self.split_tics[split]
            if not set(group["peer_TIC"].astype(str)) <= allowed:
                raise AssertionError(f"{split} TIC {anchor_tic}: cross-split peer")

    # ------------------------------------------------------------------- public
    def assert_training_ready(self) -> bool:
        """Require literal geometry for all fixed anchors before model training."""
        if any(self.excluded_anchors[split] for split in SPLITS):
            detail = {
                split: [str(self.tic[row]) for row in self.excluded_anchors[split]]
                for split in SPLITS if self.excluded_anchors[split]
            }
            raise RuntimeError(
                "spread geometry is not feasible for the unchanged anchor split; "
                f"training must not start: {detail}"
            )
        for split in SPLITS:
            rows, distances = self.peers[split]
            if rows.shape != (EXPECTED_ANCHOR_COUNTS[split], self.n_peers):
                raise AssertionError(f"{split} peer table shape drifted")
            if (rows < 0).any() or not np.isfinite(distances).all():
                raise AssertionError(f"{split} peer table contains placeholders")
        return True

    def peers_for_row(
        self, row: int, split: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the precomputed spread group for one unchanged fixed anchor."""
        row = int(row)
        if split is None:
            matches = [name for name in SPLITS if row in set(map(int, self.split_anchors[name]))]
            if len(matches) != 1:
                raise KeyError(f"row {row} is not a unique fixed anchor")
            split = matches[0]
        indices = np.flatnonzero(self.split_anchors[split] == row)
        if len(indices) != 1:
            raise KeyError(f"row {row} is not a fixed {split} anchor")
        peer_rows, distances = self.peers[split]
        selected = peer_rows[int(indices[0])]
        if (selected < 0).any():
            raise RuntimeError(f"row {row} has no feasible spread-peer group")
        return selected.copy(), distances[int(indices[0])].copy()

    def assert_peer_group(
        self, anchor: int, peer_rows: np.ndarray, distances: np.ndarray
    ) -> bool:
        """Validate the spread contract (band/angle order replaces nearest order)."""
        anchor = int(anchor)
        rows = np.asarray(peer_rows, dtype=np.int64)
        recorded = np.asarray(distances, dtype=float)
        if len(rows) != self.n_peers or len(np.unique(rows)) != self.n_peers:
            raise AssertionError("spread group is not unique-eight")
        if (self.tic[rows].astype(str) == str(self.tic[anchor])).any():
            raise AssertionError("spread group contains the anchor TIC")
        if not ((self.sector[rows] == TARGET_CHIP[0])
                & (self.camera[rows] == TARGET_CHIP[1])
                & (self.ccd[rows] == TARGET_CHIP[2])).all():
            raise AssertionError("spread peer is outside Sector 1 / camera 4 / CCD 2")
        recomputed = np.hypot(
            self.det_x[rows] - self.det_x[anchor],
            self.det_y[rows] - self.det_y[anchor],
        )
        if not np.allclose(recorded, recomputed, atol=1e-4):
            raise AssertionError("recorded spread distances disagree with coordinates")
        if not ((recomputed[:4] >= 128.0) & (recomputed[:4] < 384.0)).all():
            raise AssertionError("spread slots 1-4 violate [128,384)")
        records = self.selection_records_by_anchor.get(anchor, [])
        cap = float(records[0]["outer_radius_cap"]) if records else 768.0
        if not ((recomputed[4:] >= 384.0) & (recomputed[4:] <= cap)).all():
            raise AssertionError(f"spread slots 5-8 violate [384,{cap:g}]")
        return True

    def geometry_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "target": {"sector": 1, "camera": 4, "ccd": 2},
            "bands_px": [list(band) for band in self.spread_bands],
            "peers_per_band": self.peers_per_band,
            "spacing_ladder_px": list(self.spacing_ladder),
            "outer_boundary_expanded": bool(
                not self.peer_selection_audit.empty
                and self.peer_selection_audit["expansion_used"].any()
            ),
            "fixed_anchor_counts": {
                split: int(len(self.split_anchors[split])) for split in SPLITS
            },
            "legacy_pool_counts": {
                split: len(self.legacy_split_tics[split]) for split in SPLITS
            },
            "new_peer_tic_counts": {
                split: len(self.expanded_split_tics[split]) for split in SPLITS
            },
            "combined_pool_counts": {
                split: len(self.split_tics[split]) for split in SPLITS
            },
            "eligible_anchor_counts": {},
            "excluded_anchor_counts": {},
            "spacing_tier_anchor_counts": {},
            "unique_selected_peer_tics": int(
                self.peer_selection_audit["peer_TIC"].nunique()
                if not self.peer_selection_audit.empty else 0
            ),
        }
        for split in SPLITS:
            excluded = len(self.excluded_anchors[split])
            summary["excluded_anchor_counts"][split] = excluded
            summary["eligible_anchor_counts"][split] = (
                len(self.split_anchors[split]) - excluded
            )
            frame = self.peer_selection_audit[
                self.peer_selection_audit["split"] == split
            ] if not self.peer_selection_audit.empty else pd.DataFrame()
            counts = {}
            if not frame.empty:
                one_per_anchor = frame.drop_duplicates("anchor_TIC")
                counts = {
                    str(int(tier)): int((one_per_anchor["spacing_tier"] == tier).sum())
                    for tier in self.spacing_ladder
                }
            summary["spacing_tier_anchor_counts"][split] = counts
        if not self.peer_selection_audit.empty:
            anchor_distance = self.peer_selection_audit["anchor_distance"].to_numpy(float)
            group_min = self.peer_selection_audit.drop_duplicates(
                ["split", "anchor_TIC"]
            )["group_minimum_peer_separation"].to_numpy(float)
            summary["anchor_to_peer_distance_px"] = {
                "minimum": float(anchor_distance.min()),
                "median": float(np.median(anchor_distance)),
                "maximum": float(anchor_distance.max()),
            }
            summary["group_minimum_peer_to_peer_distance_px"] = {
                "minimum": float(group_min.min()),
                "median": float(np.median(group_min)),
                "maximum": float(group_min.max()),
            }
        return summary

    def report_spread_feasibility(self) -> None:
        summary = self.geometry_summary()
        print("spread-peer feasibility (fixed anchors; peer-only expansion):", flush=True)
        for split in SPLITS:
            print(
                f"  {split}: {summary['eligible_anchor_counts'][split]}/"
                f"{summary['fixed_anchor_counts'][split]} eligible; "
                f"new peer TICs {summary['new_peer_tic_counts'][split]}; "
                f"spacing tiers {summary['spacing_tier_anchor_counts'][split]}",
                flush=True,
            )


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (HERE.parent / path).resolve()


def build_spread_patch_from_config(
    config: dict,
    supplemental_parquet: str | Path | None = None,
    verbose: bool = True,
) -> ExpandedSpreadPeerPatch:
    """Build the fixed-anchor spread patch from a training/audit configuration."""
    base = config.get("base_parquet", config.get("parquet", DEFAULT_BASE_PARQUET))
    supplemental = (
        supplemental_parquet
        or config.get("supplemental_peer_parquet")
        or config.get("supplemental_parquet")
    )
    if supplemental is None:
        raise ValueError(
            "supplemental peer parquet is required (argument or "
            "config['supplemental_peer_parquet'])"
        )
    reference = config.get("reference_split_tics", DEFAULT_REFERENCE_SPLIT)
    return ExpandedSpreadPeerPatch(
        _repo_path(base),
        _repo_path(supplemental),
        reference_split_path=_repo_path(reference),
        curve_length=int(config.get("curve_length", 1024)),
        n_peers=int(config.get("n_peers", 8)),
        min_valid_fraction=float(config.get("min_valid_fraction", 0.5)),
        split_seed=int(config.get("seed", 42)),
        max_eligible_anchors=int(config.get("max_eligible_anchors", 400)),
        min_chip_anchors=int(config.get("min_chip_anchors", 64)),
        bands=config.get(
            "peer_distance_bands_px",
            config.get("peer_distance_bands", config.get("spread_bands", SPREAD_BANDS)),
        ),
        peers_per_band=int(config.get(
            "peers_per_distance_band", config.get("peers_per_band", 4)
        )),
        spacing_ladder=config.get(
            "minimum_peer_separation_tiers_px",
            config.get("peer_spacing_ladder_px", SPACING_LADDER),
        ),
        outer_expansion_radii=config.get("outer_expansion_radii_px", ()),
        exact_search_node_budget=int(config.get("exact_search_node_budget", 1_000_000)),
        strict_legacy_manifest=bool(config.get("strict_legacy_manifest", True)),
        verbose=verbose,
    )


__all__ = [
    "ExpandedSpreadPeerPatch",
    "SPREAD_BANDS",
    "SPACING_LADDER",
    "build_spread_patch_from_config",
    "canonical_numeric_tic",
    "stable_peer_split",
    "tic_set_sha256",
]
