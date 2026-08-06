"""Cross-sector anchor dataset: masked current sector + other sector + detector peers.

Anchors come from EVERY sector/camera/CCD chip with enough eligible stars; peers are
always drawn from the anchor's own chip, which is what keeps the cadence grid and the
detector neighbourhood shared.

One example is built around an anchor TIC `i` observed in sector `s`:

  anchor target        raw TGLC aperture flux, TIC i, sector s          [1024]
  current physics view the same curve, with cadences hidden downstream   [1024]
  cross-sector view    raw flux, TIC i, some sector != s                 [1024]
  instrument peers     eight nearest-on-detector DIFFERENT TICs, same
                       sector/camera/CCD and the same absolute cadences  [8, 1024]

Cadence grid. Each sector gets ONE absolute grid: the 1024 consecutive TGLC cadence
numbers with the most observations in that sector. A curve is placed by exact cadence
number -- never resampled, interpolated or smoothed -- so peers and anchor share the
same absolute cadences by construction. The anchor and its other-sector curve sit on
different grids (different absolute times); nothing ever compares them per cadence.

Validity. STRICT ZERO-FLAG: a cadence is valid only if time and flux are finite AND
TESS_flags == 0 AND TGLC_flags == 0. Every flagged cadence -- momentum dumps, attitude
tweaks, Argabrightening, stray light, pointing and calibration states, cosmic rays,
manual excludes -- becomes a gap. Flag values are gridded for auditing only; no branch
ever sees flagged flux. See preprocess.py.

Splits are TIC-keyed 80/10/10, so a TIC cannot leak across splits through the
cross-sector branch either, and peers are drawn only from the anchor's own split.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from disentangle_attempt.preprocess import (flag_removal_report, grid_values,
                                            preprocess_curve)

CURVE_LENGTH = 1024
N_PEERS = 8
# Peers closer than this on the detector are rejected outright: a very close neighbour
# can share the anchor's PSF wings, so its "instrument" curve carries the anchor's own
# flux and the branch could reconstruct the target from a blend rather than from shared
# measurement conditions.
PEER_MIN_DISTANCE_PX = 12.0


# ------------------------------------------------------------------ cadence grid
def build_cadence_grid(cadence_arrays, length=CURVE_LENGTH):
    """The `length` consecutive cadence numbers covering the most observations."""
    cadences = np.concatenate([np.asarray(c, np.int64) for c in cadence_arrays])
    lo, hi = int(cadences.min()), int(cadences.max())
    if hi - lo + 1 <= length:
        return np.arange(lo, lo + length, dtype=np.int64)
    counts = np.bincount(cadences - lo, minlength=hi - lo + 1).astype(np.int64)
    window = np.convolve(counts, np.ones(length, dtype=np.int64), mode="valid")
    start = lo + int(np.argmax(window))
    return np.arange(start, start + length, dtype=np.int64)


def grid_row(row, grid, length=CURVE_LENGTH):
    """One record -> (flux, valid, flagged, background, TESS flags, TGLC flags).

    Delegates to the shared `preprocess_curve`; this is the only call site, so the
    anchor target, both physics views, the peers and the quiet reference peers are all
    slices of arrays built by the identical function.
    """
    result = preprocess_curve(row["cadence_num"], row.get("time"), row["flux"],
                              row.get("TESS_flags"), row.get("TGLC_flags"), grid[:length])
    background = grid_values(row["cadence_num"], row.get("background"), grid[:length],
                             result.valid)
    return (result.curve, result.valid, result.flagged, background,
            result.tess_flags, result.tglc_flags)


# --------------------------------------------------------------------- the patch
class CrossSectorPatch:
    """Gridded curves + eligibility + TIC splits + precomputed detector peers."""

    def __init__(self, parquet_path, target_sector="auto", camera="auto", ccd="auto",
                 curve_length=CURVE_LENGTH, n_peers=N_PEERS, min_valid_fraction=0.5,
                 split_seed=42, max_eligible_anchors=None, min_chip_anchors=64,
                 require_cross_sector=False, peer_min_distance=PEER_MIN_DISTANCE_PX,
                 verbose=True):
        self.curve_length = int(curve_length)
        self.n_peers = int(n_peers)

        frame = pd.read_parquet(parquet_path)
        frame["TIC"] = frame["TIC"].astype(str)
        # A TIC can appear twice per sector when two Gaia sources resolve to it.
        frame = frame.drop_duplicates(["TIC", "sector"]).reset_index(drop=True)

        self.grids = {int(s): build_cadence_grid(g["cadence_num"], self.curve_length)
                      for s, g in frame.groupby("sector")}

        n = len(frame)
        self.X = np.zeros((n, self.curve_length), dtype=np.float32)   # normalized flux
        self.M = np.zeros((n, self.curve_length), dtype=bool)         # valid cadences
        self.Q = np.zeros((n, self.curve_length), dtype=bool)         # flagged (audit)
        self.BG = np.zeros((n, self.curve_length), dtype=np.float32)  # background
        self.F = np.zeros((n, self.curve_length), dtype=np.int64)     # TESS flags (audit)
        self.G = np.zeros((n, self.curve_length), dtype=np.int64)     # TGLC flags (audit)
        for i in range(n):
            row = frame.iloc[i]
            (self.X[i], self.M[i], self.Q[i], self.BG[i], self.F[i],
             self.G[i]) = grid_row(row, self.grids[int(row["sector"])], self.curve_length)
        # Strict zero-flag policy, enforced not assumed.
        assert not (self.M & self.Q).any(), "a flagged cadence survived as valid"
        assert not (self.M & (self.F != 0)).any(), "a TESS-flagged cadence is valid"
        assert not (self.M & (self.G != 0)).any(), "a TGLC-flagged cadence is valid"

        self.flag_report = flag_removal_report(frame)
        if verbose:
            print("cadences removed by flag type:\n"
                  + self.flag_report.to_string(index=False), flush=True)

        self.tic = frame["TIC"].to_numpy()
        self.tic_int = frame["TIC"].astype(np.int64).to_numpy()
        self.sector = frame["sector"].astype(int).to_numpy()
        self.camera = frame["camera"].astype(int).to_numpy()
        self.ccd = frame["ccd"].astype(int).to_numpy()
        self.det_x = frame["DETECTOR_X"].astype(float).to_numpy()
        self.det_y = frame["DETECTOR_Y"].astype(float).to_numpy()
        self.n_valid = self.M.sum(axis=1)
        self.min_valid = int(min_valid_fraction * self.curve_length)
        self.min_chip_anchors = int(min_chip_anchors)
        # Reproduces the pre-c74d750 eligibility rule, which also demanded a partner
        # sector. Needed to rebuild the exact TIC split an older checkpoint trained
        # with -- otherwise its test stars are not the ones it was held out from.
        self.require_cross_sector = bool(require_cross_sector)
        self.peer_min_distance = float(peer_min_distance)
        self.peer_stats = {}

        self.rows_by_tic = {}
        for i, t in enumerate(self.tic):
            self.rows_by_tic.setdefault(t, []).append(i)

        self.eligible_before_radius = self._count_eligible_ignoring_radius()
        self.chips = self._choose_target(target_sector, camera, ccd, verbose)
        self.target = self.chips[0]                     # back-compat for single-chip code
        self.chip_of_row = {}
        eligible = []
        for chip in self.chips:
            rows = self._eligible_rows(chip)
            eligible.append(rows)
            for r in rows:
                self.chip_of_row[int(r)] = chip
        self.eligible_rows = (np.concatenate(eligible) if eligible
                              else np.array([], dtype=np.int64))
        if verbose:
            print(f"training over {len(self.chips)} sector/camera/CCD chips; "
                  f"{len(self.eligible_rows)} eligible anchors total", flush=True)

        self._split(split_seed, max_eligible_anchors, verbose)

    # ------------------------------------------------------------- eligibility
    def _group_rows(self, key):
        sector, camera, ccd = key
        return np.flatnonzero((self.sector == sector) & (self.camera == camera)
                              & (self.ccd == ccd))

    def _count_eligible_ignoring_radius(self):
        """Anchors that would qualify under the old nearest-neighbour rule."""
        total = 0
        for key in {(int(a), int(b), int(c))
                    for a, b, c in zip(self.sector, self.camera, self.ccd)}:
            rows = self._group_rows(key)
            if len(rows) < self.n_peers + 1:
                continue
            keep = [i for i in rows if self.n_valid[i] >= self.min_valid]
            if self.require_cross_sector:
                keep = [i for i in keep
                        if any(self.sector[j] != key[0]
                               for j in self.rows_by_tic[self.tic[i]])]
            total += len(keep)
        return total

    def candidate_distances(self, anchor_rows, pool):
        """[len(anchor_rows), len(pool)] distances, inf where a candidate is rejected.

        Rejected: the anchor's own TIC, anything inside the exclusion radius, and any
        row without finite detector coordinates.
        """
        anchor_rows = np.atleast_1d(np.asarray(anchor_rows, dtype=np.int64))
        anchor_xy = np.stack([self.det_x[anchor_rows], self.det_y[anchor_rows]], axis=1)
        pool_xy = np.stack([self.det_x[pool], self.det_y[pool]], axis=1)
        distance = np.sqrt(((anchor_xy[:, None, :] - pool_xy[None, :, :]) ** 2).sum(-1))
        distance = np.where(self.tic[anchor_rows][:, None] == self.tic[pool][None, :],
                            np.inf, distance)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        finite_xy = np.isfinite(self.det_x[pool]) & np.isfinite(self.det_y[pool])
        distance = np.where(finite_xy[None, :], distance, np.inf)
        return np.where(distance < self.peer_min_distance, np.inf, distance)

    def _select_peers(self, anchor_rows, pool):
        """Eight nearest candidates OUTSIDE the radius, nearest first, plus a keep mask."""
        distance = self.candidate_distances(anchor_rows, pool)
        order = np.argsort(distance, axis=1)[:, :self.n_peers]
        chosen = np.take_along_axis(distance, order, axis=1)
        keep = np.isfinite(chosen).all(axis=1)          # fewer than 8 -> ineligible
        return pool[order], chosen.astype(np.float32), keep

    def _eligible_rows(self, key):
        """Rows that (1) sit on a chip with >= n_peers other TICs and (2) have enough
        valid cadences to mask and score. A second sector is NOT required: the model
        uses one curve per star (the anchor and its masked copy)."""
        rows = self._group_rows(key)
        if len(rows) < self.n_peers + 1:
            return np.array([], dtype=np.int64)
        keep = [i for i in rows if self.n_valid[i] >= self.min_valid]
        if self.require_cross_sector:
            keep = [i for i in keep
                    if any(self.sector[j] != key[0] for j in self.rows_by_tic[self.tic[i]])]
        if not keep:
            return np.array([], dtype=np.int64)
        # An anchor must have n_peers candidates outside the radius on its own chip.
        keep = np.asarray(sorted(keep), dtype=np.int64)
        usable = np.isfinite(self.candidate_distances(keep, rows)).sum(axis=1) >= self.n_peers
        return keep[usable]

    def eligibility_table(self):
        """Eligible-anchor counts per sector/camera/CCD (printed before training)."""
        keys = sorted({(int(s), int(c), int(d))
                       for s, c, d in zip(self.sector, self.camera, self.ccd)})
        return pd.DataFrame([
            {"sector": s, "camera": c, "ccd": d,
             "curves": len(self._group_rows((s, c, d))),
             "eligible_anchors": len(self._eligible_rows((s, c, d)))}
            for s, c, d in keys]).sort_values("eligible_anchors", ascending=False)

    def _choose_target(self, target_sector, camera, ccd, verbose):
        """Every sector/camera/CCD with enough eligible anchors, not just the best one.

        Peers must share the anchor's chip AND its absolute cadence grid, so a chip is
        the natural training unit. Training over many chips is what makes the
        instrument encoder see more than one detector's systematics.
        """
        table = self.eligibility_table()
        if verbose:
            print("eligible anchors by sector/camera/CCD:\n"
                  + table.head(40).to_string(index=False), flush=True)
        explicit = [v for v in (target_sector, camera, ccd) if v not in ("auto", None)]
        if len(explicit) == 3:
            return [(int(target_sector), int(camera), int(ccd))]
        if len(explicit):
            raise ValueError("set sector, camera and ccd together, or all to 'auto'")
        keep = table[table["eligible_anchors"] >= self.min_chip_anchors]
        if keep.empty:                                  # fall back to the single best
            keep = table.head(1)
        return [(int(r["sector"]), int(r["camera"]), int(r["ccd"]))
                for _, r in keep.iterrows()]

    # ------------------------------------------------------------------ splits
    def _split(self, seed, max_eligible_anchors, verbose):
        tics = np.array(sorted({self.tic[i] for i in self.eligible_rows}))
        order = np.random.default_rng(seed).permutation(len(tics))
        n_train = int(round(0.8 * len(tics)))
        n_val = int(round(0.1 * len(tics)))
        assignment = {"train": set(tics[order[:n_train]]),
                      "val": set(tics[order[n_train:n_train + n_val]]),
                      "test": set(tics[order[n_train + n_val:]])}
        assert not (assignment["train"] & assignment["val"]), "train/val TIC overlap"
        assert not (assignment["train"] & assignment["test"]), "train/test TIC overlap"
        assert not (assignment["val"] & assignment["test"]), "val/test TIC overlap"
        self.split_tics = assignment

        # One peer pool PER CHIP per split: a peer must share the anchor's chip and
        # cadence grid, so pools never mix chips.
        chip_rows = {chip: self._group_rows(chip) for chip in self.chips}
        self.split_pool = {}        # split -> chip -> candidate rows
        self.split_anchors = {}     # split -> eligible anchors (all chips)
        for name, members in assignment.items():
            pools, anchors, dropped_for_radius = {}, [], 0
            for chip in self.chips:
                pool = np.asarray([i for i in chip_rows[chip] if self.tic[i] in members],
                                  dtype=np.int64)
                if len(pool) < self.n_peers + 1:
                    continue                          # too thin in this split, skip chip
                pools[chip] = pool
                chip_anchors = np.asarray(
                    [i for i in self._eligible_rows(chip) if self.tic[i] in members],
                    dtype=np.int64)
                if len(chip_anchors):
                    _, _, keep = self._select_peers(chip_anchors, pool)
                    dropped_here = int((~keep).sum())
                    dropped_for_radius += dropped_here
                    anchors.extend(int(a) for a in chip_anchors[keep])
            anchors = np.asarray(sorted(anchors), dtype=np.int64)
            if max_eligible_anchors:                 # cap ANCHORS only; peers keep the pool
                cap = int(round(max_eligible_anchors * len(anchors)
                                / max(len(self.eligible_rows), 1)))
                if 0 < cap < len(anchors):
                    pick = np.random.default_rng(seed + 1).permutation(len(anchors))[:cap]
                    anchors = np.sort(anchors[pick])
            self.split_pool[name] = pools
            self.split_anchors[name] = anchors
            self.peer_stats[name] = {"dropped_no_valid_group": dropped_for_radius}
            if not pools:
                raise RuntimeError(f"split {name} has no chip with "
                                   f"{self.n_peers + 1} curves for peer selection")
        self.peers = {name: self._peer_table(name) for name in assignment}
        if verbose:
            self.report_peer_selection()

    def report_peer_selection(self):
        """What the exclusion radius cost, and what the surviving groups look like."""
        after = sum(len(v) for v in self.split_anchors.values())
        before = self.eligible_before_radius
        removed = (100.0 * (before - after) / before) if before else 0.0
        dropped = sum(v.get("dropped_no_valid_group", 0) for v in self.peer_stats.values())
        print(f"peer exclusion radius: {self.peer_min_distance:.1f} px", flush=True)
        print(f"  eligible anchors {before} -> {after} ({removed:.1f}% removed); "
              f"{dropped} rejected for fewer than {self.n_peers} peers outside the radius",
              flush=True)
        for name in ("train", "val", "test"):
            distances = self.peers[name][1]
            if not len(distances):
                continue
            print(f"  {name}: {len(self.split_anchors[name])} anchors over "
                  f"{len(self.split_pool[name])} chips | selected peer distance "
                  f"median {np.median(distances):.2f} px, min {distances.min():.2f}, "
                  f"max {distances.max():.2f}", flush=True)

    def assert_peer_group(self, anchor, peer_rows, distances):
        """Every contract the peer rule promises, checked on one group."""
        anchor = int(anchor)
        peer_rows = np.asarray(peer_rows, dtype=np.int64)
        distances = np.asarray(distances, dtype=float)
        assert len(peer_rows) == self.n_peers == len(distances), "group is not n_peers"
        assert (distances >= self.peer_min_distance - 1e-6).all(), \
            f"a peer is inside the {self.peer_min_distance} px exclusion radius"
        assert (np.diff(distances) >= -1e-6).all(), "peers are not nearest-to-farthest"
        assert (self.tic[peer_rows] != self.tic[anchor]).all(), "a peer is the anchor TIC"
        assert (self.sector[peer_rows] == self.sector[anchor]).all(), "peer sector differs"
        assert (self.camera[peer_rows] == self.camera[anchor]).all(), "peer camera differs"
        assert (self.ccd[peer_rows] == self.ccd[anchor]).all(), "peer CCD differs"
        recomputed = np.sqrt((self.det_x[peer_rows] - self.det_x[anchor]) ** 2
                             + (self.det_y[peer_rows] - self.det_y[anchor]) ** 2)
        assert np.allclose(recomputed, distances, atol=1e-4), "distances disagree"
        return True

    def _peer_table(self, split):
        """Eight nearest DIFFERENT TICs outside the exclusion radius, own chip only."""
        anchors = self.split_anchors[split]
        rows_out = np.zeros((len(anchors), self.n_peers), np.int64)
        dist_out = np.zeros((len(anchors), self.n_peers), np.float32)
        if len(anchors) == 0:
            return rows_out, dist_out
        by_chip = {}
        for index, anchor in enumerate(anchors):
            by_chip.setdefault(self.chip_of_row[int(anchor)], []).append(index)
        for chip, indices in by_chip.items():
            pool = self.split_pool[split][chip]
            rows, chosen, keep = self._select_peers(anchors[indices], pool)
            assert keep.all(), f"chip {chip}: an anchor kept without a full peer group"
            rows_out[indices] = rows
            dist_out[indices] = chosen
        assert (dist_out >= self.peer_min_distance - 1e-6).all(), \
            "a selected peer is inside the exclusion radius"
        assert (np.diff(dist_out, axis=1) >= -1e-6).all(), "peers are not distance-ordered"
        return rows_out.astype(np.int64), dist_out.astype(np.float32)

    # -------------------------------------------------------------- accessors
    def other_sector_rows(self, row):
        return [j for j in self.rows_by_tic[self.tic[row]] if self.sector[j] != self.sector[row]]

    def split_of_tic(self, tic):
        for name, members in self.split_tics.items():
            if tic in members:
                return name
        raise KeyError(f"TIC {tic} is not an eligible anchor in any split")

    def peers_for_row(self, row, split=None):
        """Eight nearest different-TIC peers outside the radius, within its chip."""
        split = split or self.split_of_tic(self.tic[row])
        chip = self.chip_of_row.get(int(row),
                                    (int(self.sector[row]), int(self.camera[row]),
                                     int(self.ccd[row])))
        pool = self.split_pool[split][chip]
        rows, chosen, keep = self._select_peers(np.asarray([int(row)]), pool)
        assert keep[0], (f"row {row}: fewer than {self.n_peers} peers outside "
                         f"{self.peer_min_distance} px")
        return rows[0], chosen[0]

    def row_for_tic(self, tic):
        rows = [i for i in self.rows_by_tic[str(tic)] if int(i) in self.chip_of_row]
        if not rows:
            raise KeyError(f"TIC {tic} is not an eligible anchor on any trained chip")
        return int(rows[0])

    def random_peer_rows(self, anchor_rows, split, rng):
        """Random peers from the anchor's OWN chip (branch-use control).

        Same sector/camera/CCD and cadence grid as the true peers -- only detector
        proximity differs, which is exactly what the control isolates.
        """
        out = np.zeros((len(anchor_rows), self.n_peers), dtype=np.int64)
        for k, anchor in enumerate(anchor_rows):
            chip = self.chip_of_row[int(anchor)]
            pool = self.split_pool[split][chip]
            # Same candidate pool as the real peers -- radius included -- so the control
            # differs only in proximity, not in which stars are admissible.
            allowed = pool[np.isfinite(
                self.candidate_distances(np.asarray([int(anchor)]), pool)[0])]
            out[k] = rng.choice(allowed, size=self.n_peers, replace=False)
        return out

    def curves(self, rows):
        return (torch.from_numpy(self.X[rows]), torch.from_numpy(self.M[rows]))


class CrossSectorAnchorDataset(Dataset):
    """One item = one anchor bundle; a batch of 32 gives the step tensors.

    One curve per star: the anchor is both the target and (after masking) the physics
    input. No partner sector is loaded."""

    def __init__(self, patch, split, seed=0):
        self.patch, self.split = patch, split
        self.anchors = patch.split_anchors[split]
        self.peer_rows, self.peer_distance = patch.peers[split]
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        """Kept so the training loop can vary per-epoch sampling."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        patch = self.patch
        anchor = int(self.anchors[index])
        peers = self.peer_rows[index]

        return {
            "anchor_raw": torch.from_numpy(patch.X[anchor]),
            "anchor_valid_mask": torch.from_numpy(patch.M[anchor]),
            "peer_raw": torch.from_numpy(patch.X[peers]),
            "peer_mask": torch.from_numpy(patch.M[peers]),
            "anchor_tic_ids": torch.tensor(patch.tic_int[anchor], dtype=torch.int64),
            "anchor_sector": torch.tensor(patch.sector[anchor], dtype=torch.int64),
            "peer_tic_ids": torch.from_numpy(patch.tic_int[peers]),
            "peer_distances": torch.from_numpy(self.peer_distance[index]),
            "anchor_row": torch.tensor(anchor, dtype=torch.int64),
            "peer_rows": torch.from_numpy(np.asarray(peers, dtype=np.int64)),
        }


def target_from_checkpoint(state, config):
    """Rebuild the SAME chip set the checkpoint trained on.

    checkpoint["target"] is only chips[0], a legacy single-chip field. When the config
    asked for 'auto' the run covered every qualifying chip, so honouring `target` here
    would rebuild one chip -- and a different TIC split, making the split labels wrong.
    """
    configured = (config.get("sector", "auto"), config.get("camera", "auto"),
                  config.get("ccd", "auto"))
    if all(v not in ("auto", None) for v in configured):
        return configured                       # the run pinned a chip explicitly
    if "physics_consistency_weight" in config:
        # Pre-c74d750 code auto-picked exactly ONE chip, recorded in `target`.
        return state.get("target") or ("auto", "auto", "auto")
    return "auto", "auto", "auto"               # multi-chip run: every qualifying chip


def infer_require_cross_sector(config, override="auto"):
    """Which eligibility rule did this checkpoint train under?

    Runs before commit ad6c7ad required a partner sector; the tell is that their config
    still carries physics_consistency_weight. Getting this wrong rebuilds a different
    TIC split, so "held-out" stars would not be the ones the model was held out from.
    """
    if override in ("yes", True):
        return True
    if override in ("no", False):
        return False
    return "physics_consistency_weight" in config


def audit_batch(patch, batch, verbose=True):
    """Assert the data contract on a real batch, and print row 0.

    Peers sit on the anchor's sector/camera/CCD with different TICs, ordered by
    detector distance, and nothing flagged is model-visible.
    """
    anchor_rows = batch["anchor_row"].numpy()
    peer_rows = batch["peer_rows"].numpy()

    distances = batch["peer_distances"].numpy()
    for k, anchor in enumerate(anchor_rows):
        patch.assert_peer_group(anchor, peer_rows[k], distances[k])
    assert np.allclose(batch["anchor_raw"].numpy(), patch.X[anchor_rows]), \
        "anchor curve was altered before the model"

    for name, rows in (("anchor", anchor_rows), ("peers", peer_rows.reshape(-1))):
        assert not (patch.M[rows] & (patch.F[rows] != 0)).any(), f"{name}: TESS-flagged cadence valid"
        assert not (patch.M[rows] & (patch.G[rows] != 0)).any(), f"{name}: TGLC-flagged cadence valid"

    if verbose:
        a = int(anchor_rows[0])
        peers = peer_rows[0]
        print("--- audited batch (row 0) ---")
        print(f"  anchor TIC {patch.tic[a]}  sector {patch.sector[a]}  "
              f"cam{patch.camera[a]}-ccd{patch.ccd[a]}  "
              f"det ({patch.det_x[a]:.1f}, {patch.det_y[a]:.1f})  valid {patch.n_valid[a]}")
        print(f"  peers (same sector/camera/CCD, all >= "
              f"{patch.peer_min_distance:.1f} px), nearest first:")
        for row, dist in zip(peers, distances[0]):
            print(f"     TIC {patch.tic[row]:>12s}  det ({patch.det_x[row]:7.1f}, "
                  f"{patch.det_y[row]:7.1f})  distance {dist:8.3f} px")
        print("-----------------------------", flush=True)
    return True


def _audit(parquet_path, curve_length=CURVE_LENGTH, n_peers=N_PEERS):
    """Can this parquet support the experiment at all?

        python -m disentangle_attempt.dataset <parquet>

    Prints eligible cross-sector anchors per sector/camera/CCD. A parquet whose stars
    were sampled independently per sector has ~zero eligible anchors: the SAME TIC has
    to appear in two sectors, which sky-uniform per-sector sampling almost never gives.
    """
    frame = pd.read_parquet(parquet_path, columns=["TIC", "sector"])
    frame["TIC"] = frame["TIC"].astype(str)
    per_tic = frame.drop_duplicates(["TIC", "sector"]).groupby("TIC")["sector"].nunique()
    multi = int((per_tic >= 2).sum())
    print(f"{parquet_path}\n  {len(frame)} rows, {len(per_tic)} TICs, "
          f"{multi} TICs observed in >= 2 sectors")
    if multi < n_peers + 1:
        print("  -> NOT usable: run disentangle_attempt.fetch_data to build a "
              "cross-sector patch first")
        return
    patch = CrossSectorPatch(parquet_path, curve_length=curve_length, n_peers=n_peers,
                             verbose=False)
    print(patch.eligibility_table().to_string(index=False))


if __name__ == "__main__":
    import sys
    _audit(sys.argv[1] if len(sys.argv) > 1
           else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "cross_sector_raw.parquet"))
