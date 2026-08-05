"""Cross-sector anchor dataset: masked current sector + other sector + detector peers.

One example is built around an anchor TIC `i` observed in target sector `s`:

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
                 split_seed=42, max_eligible_anchors=None, verbose=True):
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

        self.rows_by_tic = {}
        for i, t in enumerate(self.tic):
            self.rows_by_tic.setdefault(t, []).append(i)

        self.target = self._choose_target(target_sector, camera, ccd, verbose)
        self.eligible_rows = self._eligible_rows(self.target)
        if verbose:
            print(f"target sector/camera/CCD = {self.target}; "
                  f"{len(self.eligible_rows)} eligible cross-sector anchors", flush=True)

        self._split(split_seed, max_eligible_anchors, verbose)

    # ------------------------------------------------------------- eligibility
    def _group_rows(self, key):
        sector, camera, ccd = key
        return np.flatnonzero((self.sector == sector) & (self.camera == camera)
                              & (self.ccd == ccd))

    def _eligible_rows(self, key):
        """Rows that (1) have the same TIC in another sector, (2) sit on a chip with
        >= n_peers other TICs, and (3) have enough valid cadences to mask and score."""
        rows = self._group_rows(key)
        if len(rows) < self.n_peers + 1:
            return np.array([], dtype=np.int64)
        keep = [i for i in rows
                if self.n_valid[i] >= self.min_valid
                and any(self.sector[j] != key[0] for j in self.rows_by_tic[self.tic[i]])]
        return np.asarray(sorted(keep), dtype=np.int64)

    def eligibility_table(self):
        """Eligible-anchor counts per sector/camera/CCD (printed before training)."""
        keys = sorted({(int(s), int(c), int(d))
                       for s, c, d in zip(self.sector, self.camera, self.ccd)})
        return pd.DataFrame([
            {"sector": s, "camera": c, "ccd": d,
             "curves": len(self._group_rows((s, c, d))),
             "eligible_tics": len(self._eligible_rows((s, c, d)))}
            for s, c, d in keys]).sort_values("eligible_tics", ascending=False)

    def _choose_target(self, target_sector, camera, ccd, verbose):
        table = self.eligibility_table()
        if verbose:
            print("eligible cross-sector TICs by sector/camera/CCD:\n"
                  + table.to_string(index=False), flush=True)
        explicit = [v for v in (target_sector, camera, ccd) if v not in ("auto", None)]
        if len(explicit) == 3:
            return (int(target_sector), int(camera), int(ccd))
        if len(explicit):
            raise ValueError("set sector, camera and ccd together, or all to 'auto'")
        best = table.iloc[0]
        return (int(best["sector"]), int(best["camera"]), int(best["ccd"]))

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

        group = self._group_rows(self.target)
        self.split_pool = {}        # every chip row of the split -> peer candidates
        self.split_anchors = {}     # eligible anchors of the split
        for name, members in assignment.items():
            self.split_pool[name] = np.asarray(
                [i for i in group if self.tic[i] in members], dtype=np.int64)
            anchors = np.asarray(
                [i for i in self.eligible_rows if self.tic[i] in members], dtype=np.int64)
            if max_eligible_anchors:                 # cap ANCHORS only; peers keep the pool
                cap = int(round(max_eligible_anchors * len(anchors) / max(len(self.eligible_rows), 1)))
                if 0 < cap < len(anchors):
                    pick = np.random.default_rng(seed + 1).permutation(len(anchors))[:cap]
                    anchors = np.sort(anchors[pick])
            self.split_anchors[name] = anchors
            if len(self.split_pool[name]) < self.n_peers + 1:
                raise RuntimeError(f"split {name} has only {len(self.split_pool[name])} "
                                   f"chip rows; need {self.n_peers + 1} for peer selection")
        self.peers = {name: self._peer_table(name) for name in assignment}
        if verbose:
            for name in ("train", "val", "test"):
                print(f"  {name}: {len(self.split_anchors[name])} anchors "
                      f"from a {len(self.split_pool[name])}-curve peer pool", flush=True)

    def _peer_table(self, split):
        """Eight nearest DIFFERENT TICs on the detector, within the split's pool."""
        anchors, pool = self.split_anchors[split], self.split_pool[split]
        if len(anchors) == 0:
            return np.zeros((0, self.n_peers), np.int64), np.zeros((0, self.n_peers), np.float32)
        pool_xy = np.stack([self.det_x[pool], self.det_y[pool]], axis=1)
        anchor_xy = np.stack([self.det_x[anchors], self.det_y[anchors]], axis=1)
        distance = np.sqrt(((anchor_xy[:, None, :] - pool_xy[None, :, :]) ** 2).sum(-1))
        same_tic = self.tic[anchors][:, None] == self.tic[pool][None, :]
        distance = np.where(same_tic, np.inf, distance)          # never the anchor TIC
        order = np.argsort(distance, axis=1)[:, :self.n_peers]
        rows = pool[order]
        chosen = np.take_along_axis(distance, order, axis=1)
        assert np.isfinite(chosen).all(), "fewer than n_peers different TICs available"
        return rows.astype(np.int64), chosen.astype(np.float32)

    # -------------------------------------------------------------- accessors
    def other_sector_rows(self, row):
        return [j for j in self.rows_by_tic[self.tic[row]] if self.sector[j] != self.sector[row]]

    def split_of_tic(self, tic):
        for name, members in self.split_tics.items():
            if tic in members:
                return name
        raise KeyError(f"TIC {tic} is not an eligible anchor in any split")

    def peers_for_row(self, row, split=None):
        """Eight nearest different-TIC peers for an arbitrary row of the target chip."""
        split = split or self.split_of_tic(self.tic[row])
        pool = self.split_pool[split]
        distance = np.sqrt((self.det_x[pool] - self.det_x[row]) ** 2
                           + (self.det_y[pool] - self.det_y[row]) ** 2)
        distance = np.where(self.tic[pool] == self.tic[row], np.inf, distance)
        order = np.argsort(distance)[:self.n_peers]
        assert np.isfinite(distance[order]).all(), "fewer than n_peers different TICs"
        return pool[order], distance[order].astype(np.float32)

    def row_for_tic(self, tic):
        sector = self.target[0]
        rows = [i for i in self.rows_by_tic[str(tic)] if self.sector[i] == sector]
        if not rows:
            raise KeyError(f"TIC {tic} has no curve in target sector {sector}")
        return int(rows[0])

    def random_peer_rows(self, anchor_rows, split, rng):
        """Random same-sector/camera/CCD peers from the split (branch-use control)."""
        pool = self.split_pool[split]
        out = np.zeros((len(anchor_rows), self.n_peers), dtype=np.int64)
        for k, anchor in enumerate(anchor_rows):
            candidates = pool[self.tic[pool] != self.tic[anchor]]
            out[k] = rng.choice(candidates, size=self.n_peers, replace=False)
        return out

    def curves(self, rows):
        return (torch.from_numpy(self.X[rows]), torch.from_numpy(self.M[rows]))


class CrossSectorAnchorDataset(Dataset):
    """One item = one anchor bundle; a batch of 32 gives the step tensors."""

    def __init__(self, patch, split, seed=0):
        self.patch, self.split = patch, split
        self.anchors = patch.split_anchors[split]
        self.peer_rows, self.peer_distance = patch.peers[split]
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        """The alternative sector is redrawn each epoch when a TIC has several."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        patch = self.patch
        anchor = int(self.anchors[index])
        others = patch.other_sector_rows(anchor)
        rng = np.random.default_rng((self.seed, self.epoch, anchor))
        other = int(others[rng.integers(len(others))])
        peers = self.peer_rows[index]

        return {
            "anchor_raw": torch.from_numpy(patch.X[anchor]),
            "anchor_valid_mask": torch.from_numpy(patch.M[anchor]),
            "other_sector_raw": torch.from_numpy(patch.X[other]),
            "other_sector_mask": torch.from_numpy(patch.M[other]),
            "peer_raw": torch.from_numpy(patch.X[peers]),
            "peer_mask": torch.from_numpy(patch.M[peers]),
            "anchor_tic_ids": torch.tensor(patch.tic_int[anchor], dtype=torch.int64),
            "anchor_sector": torch.tensor(patch.sector[anchor], dtype=torch.int64),
            "other_sector": torch.tensor(patch.sector[other], dtype=torch.int64),
            "peer_tic_ids": torch.from_numpy(patch.tic_int[peers]),
            "peer_distances": torch.from_numpy(self.peer_distance[index]),
            "anchor_row": torch.tensor(anchor, dtype=torch.int64),
            "other_row": torch.tensor(other, dtype=torch.int64),
            "peer_rows": torch.from_numpy(np.asarray(peers, dtype=np.int64)),
        }


def audit_batch(patch, batch, verbose=True):
    """Assert the direct cross-sector contract on a real batch, and print row 0.

    physics TIC == anchor TIC, physics sector != anchor sector, peers on the anchor's
    sector/camera/CCD with different TICs, ordered by detector distance, and no
    anchor-sector flux anywhere in the physics input.
    """
    anchor_rows = batch["anchor_row"].numpy()
    other_rows = batch["other_row"].numpy()
    peer_rows = batch["peer_rows"].numpy()

    assert (patch.tic_int[anchor_rows] == patch.tic_int[other_rows]).all(), \
        "physics curve is a different TIC than the anchor"
    assert (patch.sector[anchor_rows] != patch.sector[other_rows]).all(), \
        "physics curve is from the anchor's own sector"
    for k, anchor in enumerate(anchor_rows):
        peers = peer_rows[k]
        assert (patch.sector[peers] == patch.sector[anchor]).all(), "peer sector differs"
        assert (patch.camera[peers] == patch.camera[anchor]).all(), "peer camera differs"
        assert (patch.ccd[peers] == patch.ccd[anchor]).all(), "peer CCD differs"
        assert (patch.tic_int[peers] != patch.tic_int[anchor]).all(), "a peer is the anchor TIC"
    distances = batch["peer_distances"].numpy()
    assert (np.diff(distances, axis=1) >= -1e-6).all(), "peers are not distance-ordered"
    recomputed = np.sqrt((patch.det_x[peer_rows] - patch.det_x[anchor_rows][:, None]) ** 2
                         + (patch.det_y[peer_rows] - patch.det_y[anchor_rows][:, None]) ** 2)
    assert np.allclose(recomputed, distances, atol=1e-4), "stored distances disagree"

    # No anchor-sector flux may enter the physics encoder.
    physics = batch["other_sector_raw"].numpy()
    assert np.allclose(physics, patch.X[other_rows]), "physics input is not the other-sector curve"
    for k, anchor in enumerate(anchor_rows):
        if patch.n_valid[anchor] > 0:
            assert not np.array_equal(physics[k], patch.X[anchor]), \
                "physics input equals the anchor-sector curve"
    # Quality policy: nothing flagged is model-visible.
    for name, rows in (("anchor", anchor_rows), ("physics", other_rows),
                       ("peers", peer_rows.reshape(-1))):
        assert not (patch.M[rows] & (patch.F[rows] != 0)).any(), f"{name}: TESS-flagged cadence valid"
        assert not (patch.M[rows] & (patch.G[rows] != 0)).any(), f"{name}: TGLC-flagged cadence valid"

    if verbose:
        a, o = int(anchor_rows[0]), int(other_rows[0])
        peers = peer_rows[0]
        print("--- audited batch (row 0) ---")
        print(f"  anchor        TIC {patch.tic[a]}  sector {patch.sector[a]}  "
              f"cam{patch.camera[a]}-ccd{patch.ccd[a]}  "
              f"det ({patch.det_x[a]:.1f}, {patch.det_y[a]:.1f})  valid {patch.n_valid[a]}")
        print(f"  physics input TIC {patch.tic[o]}  sector {patch.sector[o]}  "
              f"cam{patch.camera[o]}-ccd{patch.ccd[o]}  valid {patch.n_valid[o]}")
        print(f"  peers (sector {patch.sector[peers[0]]}, "
              f"cam{patch.camera[peers[0]]}-ccd{patch.ccd[peers[0]]}), nearest first:")
        for tic, dist in zip(patch.tic[peers], distances[0]):
            print(f"     TIC {tic:>12s}   distance {dist:8.3f} px")
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
