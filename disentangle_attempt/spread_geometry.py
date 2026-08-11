"""The two-band spread-peer selection rule, isolated from any one chip.

This is a verbatim port of the geometry in
:class:`disentangle_attempt.spread_peers.ExpandedSpreadPeerPatch`, lifted out so a
multi-chip patch can apply the identical rule per chip.  ``spread_peers.py`` itself is
deliberately left untouched: it carries the frozen legacy manifest of the completed
Sector-1 / camera-4 / CCD-2 run, and that run must stay reproducible.

``test_spread_geometry_port.py`` checks the port against that run's audited
``peer_selection.csv``, so "identical rule" is verified rather than asserted.

The rule, unchanged:

* slots 1--4 lie in the half-open inner band ``[128, 384)`` pixels from the anchor;
* slots 5--8 lie in the inclusive outer band ``[384, 768]`` pixels;
* every selected pair of peers is at least 256 px apart, relaxed to 192 then 128 px
  only after the whole chip fails at the stricter tier;
* only after that complete ladder may the outer upper boundary expand, keeping the
  already-relaxed 128 px spacing;
* peers are different TICs from the anchor and from each other, and angular coverage
  around the anchor is maximized before pairwise spread breaks ties.

Detector coordinates are used only to choose peers.  They never become model inputs,
targets, or a constructed correction curve.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


SPREAD_BANDS = ((128.0, 384.0), (384.0, 768.0))
SPACING_LADDER = (256.0, 192.0, 128.0)
PEERS_PER_BAND = 4
N_PEERS = 8


def circular_max_gap(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 2.0 * np.pi
    ordered = np.sort(np.mod(np.asarray(angles, dtype=float), 2.0 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    return float(gaps.max())


class SpreadPeerSelector:
    """Applies the two-band, minimum-separation spread rule to one anchor at a time.

    The selector holds only coordinate and identity arrays, so the caller decides what a
    "pool" is.  For a multi-chip data set the pool must be the anchor's own chip and own
    split, which is what keeps the cadence grid shared and the TIC split leak-free.
    """

    def __init__(
        self,
        det_x: np.ndarray,
        det_y: np.ndarray,
        tic: np.ndarray,
        tic_int: np.ndarray,
        *,
        bands: Sequence[Sequence[float]] = SPREAD_BANDS,
        peers_per_band: int = PEERS_PER_BAND,
        spacing_ladder: Sequence[float] = SPACING_LADDER,
        outer_expansion_radii: Sequence[float] = (),
        n_peers: int = N_PEERS,
        exact_search_node_budget: int = 1_000_000,
    ):
        self.det_x = np.asarray(det_x, dtype=float)
        self.det_y = np.asarray(det_y, dtype=float)
        self.tic = np.asarray(tic)
        self.tic_int = np.asarray(tic_int, dtype=np.int64)
        self.spread_bands = tuple(tuple(float(v) for v in band) for band in bands)
        self.peers_per_band = int(peers_per_band)
        self.spacing_ladder = tuple(float(v) for v in spacing_ladder)
        self.outer_expansion_radii = tuple(float(v) for v in outer_expansion_radii)
        self.n_peers = int(n_peers)
        self.exact_search_node_budget = int(exact_search_node_budget)

        if self.spread_bands != SPREAD_BANDS:
            raise ValueError(f"distance bands must remain literal {SPREAD_BANDS}")
        if self.peers_per_band != PEERS_PER_BAND:
            raise ValueError("the rule requires four peers in each band")
        if self.spacing_ladder != SPACING_LADDER:
            raise ValueError(f"spacing ladder must remain literal {SPACING_LADDER}")
        if self.n_peers != self.peers_per_band * 2:
            raise ValueError("n_peers must be twice the per-band count")
        previous_cap = self.spread_bands[-1][1]
        for cap in self.outer_expansion_radii:
            if cap <= previous_cap:
                raise ValueError("outer expansion radii must increase beyond 768 pixels")
            previous_cap = cap

    # ------------------------------------------------------------- candidate bands
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
        inside = (distance >= low) & (
            distance < high if band_index == 0 else distance <= high
        )
        rows = rows[different & finite & inside]
        if not len(rows):
            return rows
        dx = self.det_x[rows] - self.det_x[anchor]
        dy = self.det_y[rows] - self.det_y[anchor]
        distance = np.hypot(dx, dy)
        angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        order = np.lexsort((self.tic_int[rows], distance, angle))
        return rows[order]

    def _compatible(
        self, rows: np.ndarray, selected: list[int], spacing: float
    ) -> np.ndarray:
        if not selected or not len(rows):
            return np.ones(len(rows), dtype=bool)
        xy = np.column_stack([self.det_x[rows], self.det_y[rows]])
        chosen_xy = np.column_stack(
            [
                self.det_x[np.asarray(selected, dtype=np.int64)],
                self.det_y[np.asarray(selected, dtype=np.int64)],
            ]
        )
        return (
            np.linalg.norm(xy[:, None, :] - chosen_xy[None, :, :], axis=2)
            >= spacing - 1e-9
        ).all(axis=1)

    # ------------------------------------------------------- multi-start greedy FPS
    def _seed_rows(self, candidates: tuple[np.ndarray, np.ndarray]) -> list[int]:
        seeds: list[int] = []
        for rows in candidates:
            if not len(rows):
                continue
            count = min(24, len(rows))
            indices = np.unique(
                np.linspace(0, len(rows) - 1, count).round().astype(int)
            )
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
            band = next(
                (value for value in schedule if counts[value] < self.peers_per_band),
                None,
            )
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
                available = candidates[other][
                    self._compatible(candidates[other], selected, spacing)
                ]
                available = available[~np.isin(available, np.asarray(selected))]
                if not len(available):
                    return None
                band = other
            chosen_xy = np.column_stack(
                [self.det_x[np.asarray(selected)], self.det_y[np.asarray(selected)]]
            )
            xy = np.column_stack([self.det_x[available], self.det_y[available]])
            min_space = np.linalg.norm(xy[:, None] - chosen_xy[None], axis=2).min(axis=1)
            angle = np.mod(
                np.arctan2(
                    self.det_y[available] - self.det_y[anchor],
                    self.det_x[available] - self.det_x[anchor],
                ),
                2.0 * np.pi,
            )
            selected_angle = np.mod(
                np.arctan2(
                    self.det_y[np.asarray(selected)] - self.det_y[anchor],
                    self.det_x[np.asarray(selected)] - self.det_x[anchor],
                ),
                2.0 * np.pi,
            )
            delta = np.abs(angle[:, None] - selected_angle[None])
            min_angle = np.minimum(delta, 2.0 * np.pi - delta).min(axis=1)
            anchor_distance = np.hypot(
                self.det_x[available] - self.det_x[anchor],
                self.det_y[available] - self.det_y[anchor],
            )
            # np.lexsort uses the final key as primary.  Small TIC is the final,
            # deterministic tie breaker after spatial and angular coverage.
            order = np.lexsort(
                (self.tic_int[available], -anchor_distance, -min_angle, -min_space)
            )
            selected.append(int(available[order[0]]))
            counts[band] += 1
        return (
            selected
            if counts == [self.peers_per_band, self.peers_per_band]
            else None
        )

    def _solution_score(self, anchor: int, rows: list[int]) -> tuple[Any, ...]:
        selected = np.asarray(rows, dtype=np.int64)
        xy = np.column_stack([self.det_x[selected], self.det_y[selected]])
        pair = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        triangle = pair[np.triu_indices(len(selected), 1)]
        angles = np.mod(
            np.arctan2(
                self.det_y[selected] - self.det_y[anchor],
                self.det_x[selected] - self.det_x[anchor],
            ),
            2.0 * np.pi,
        )
        # Angular coverage is primary; pairwise spatial spread breaks ties.
        return (
            -circular_max_gap(angles),
            -circular_max_gap(angles[: self.peers_per_band]),
            -circular_max_gap(angles[self.peers_per_band :]),
            float(triangle.min()),
            float(triangle.sum()),
            tuple(-int(self.tic_int[row]) for row in selected),
        )

    def _band_angle_order(
        self, anchor: int, solution: list[int], candidates: tuple[np.ndarray, np.ndarray]
    ) -> list[int]:
        """Public slot order is band then angle, irrespective of search traversal."""
        ordered: list[int] = []
        for band_rows in candidates:
            band_set = set(map(int, band_rows))
            chosen = [row for row in solution if row in band_set]
            chosen.sort(
                key=lambda row: (
                    float(
                        np.mod(
                            np.arctan2(
                                self.det_y[row] - self.det_y[anchor],
                                self.det_x[row] - self.det_x[anchor],
                            ),
                            2.0 * np.pi,
                        )
                    ),
                    float(
                        np.hypot(
                            self.det_x[row] - self.det_x[anchor],
                            self.det_y[row] - self.det_y[anchor],
                        )
                    ),
                    int(self.tic_int[row]),
                )
            )
            ordered.extend(chosen)
        return ordered

    def _greedy_fps(
        self, anchor: int, candidates: tuple[np.ndarray, np.ndarray], spacing: float
    ) -> list[int] | None:
        solutions: list[list[int]] = []
        for seed in self._seed_rows(candidates):
            selected = self._greedy_from_seed(anchor, candidates, spacing, seed)
            if selected is None:
                continue
            ordered = self._band_angle_order(anchor, selected, candidates)
            if len(ordered) == self.n_peers:
                solutions.append(ordered)
        return (
            max(solutions, key=lambda rows: self._solution_score(anchor, rows))
            if solutions
            else None
        )

    # --------------------------------------------------- bounded exact backtracking
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
                return (
                    list(picked) + outer_solution if outer_solution is not None else None
                )
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

    # --------------------------------------------------------------- public entry
    def select_for_anchor(
        self, anchor: int, pool: np.ndarray
    ) -> tuple[list[int] | None, list[dict[str, Any]]]:
        """Return the eight peer rows in audited slot order, plus every attempt made."""
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
                return self._band_angle_order(anchor, solution, candidates), attempts
            attempt["status"] = "infeasible_exhaustive_search"
            attempt["relaxation_reason"] = "no feasible unique-eight group at this stage"
            attempts.append(attempt)
        return None, attempts


__all__ = [
    "SPREAD_BANDS",
    "SPACING_LADDER",
    "PEERS_PER_BAND",
    "N_PEERS",
    "SpreadPeerSelector",
    "circular_max_gap",
]
