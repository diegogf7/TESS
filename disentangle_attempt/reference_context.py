"""Pick one OBSERVED eight-peer group per sector/camera/CCD to serve as the quiet
instrument context at inference.

The cleaned curve is a counterfactual: "what would the decoder predict for this star
if its detector neighbourhood had been as quiet as the quietest group we actually
observed on this chip". Nothing here is a correction target, and no zero/synthetic
instrument latent is ever fabricated -- the substituted context is real data.

Selection:
  1. candidates come only from the TRAIN split;
  2. every candidate sits on the same absolute 1024-cadence grid as the target
     (same sector/camera/CCD -- asserted, not assumed);
  3. a seed row supplies a detector location only; its eight nearest different-TIC
     neighbours OUTSIDE the peer exclusion radius form the context, ordered by distance
     from the seed, and the seed itself is never encoded;
  4. every valid cadence is already unflagged (strict zero-flag preprocessing), so no
     flagged cadence can contribute to the score; groups with too few valid cadences
     are rejected;
  5. the ranking uses the already-available background column -- the robust
     amplitude of the group-median background -- falling back to the robust
     amplitude of the group-median flux after per-star medians are removed;
  6. the lowest-scoring group wins.
"""

import json
import os

import numpy as np
import torch

# preprocess_curve removes every flagged cadence, so `flagged & valid` is empty by
# construction and this threshold can never bind. It is retained as a tripwire: a
# nonzero fraction here would mean the policy regressed.
MAX_SYSTEMATIC_FRACTION = 0.0


def robust_amplitude(values):
    """1.4826 * MAD -- the scale the repository uses everywhere for raw flux."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("inf")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _group_score(patch, rows, use_background):
    """Lower = quieter. Scored only on cadences the group observes and that carry no
    systematic-event flag, so a flagged excursion never masquerades as quiet."""
    valid = patch.M[rows]                                     # [8, L]
    events = patch.Q[rows] & valid                            # empty under the policy
    usable = valid.sum(axis=0) >= len(rows) // 2
    if usable.sum() < 0.25 * patch.curve_length:
        return float("inf"), 0.0, "insufficient-usable-cadences"

    event_fraction = float(events[valid].mean()) if valid.any() else 0.0
    if event_fraction > MAX_SYSTEMATIC_FRACTION:
        raise RuntimeError("flagged cadences are marked valid -- preprocessing regressed")

    if use_background:
        background = patch.BG[rows][:, usable]
        curve = np.median(np.where(valid[:, usable], background, np.nan), axis=0)
        basis = "background"
    else:
        flux = patch.X[rows][:, usable].astype(np.float64)
        flux = flux - np.nanmedian(np.where(valid[:, usable], flux, np.nan),
                                   axis=1, keepdims=True)     # remove stellar medians
        curve = np.nanmedian(np.where(valid[:, usable], flux, np.nan), axis=0)
        basis = "group-median-flux"
    curve = curve[np.isfinite(curve)]
    return robust_amplitude(curve), event_fraction, basis


def build_reference_context(patch, split="train", n_peers=None, chip=None, verbose=True):
    """Scan every train-split seed location on ONE chip and return its quietest group.

    A quiet reference is chip-specific by construction: its peers must share the target's
    absolute cadence grid and detector neighbourhood, so a multi-chip run needs one
    reference per chip, not one overall.
    """
    n_peers = int(n_peers or patch.n_peers)
    chip = tuple(chip) if chip is not None else patch.chips[0]
    pool = patch.split_pool[split][chip]
    sector, camera, ccd = chip
    assert (patch.sector[pool] == sector).all() and (patch.camera[pool] == camera).all() \
        and (patch.ccd[pool] == ccd).all(), "candidate pool is off the target chip"

    # Same candidate rule as training: different TIC, same chip, outside the radius.
    distance = patch.candidate_distances(pool, pool)

    use_background = bool(np.isfinite(patch.BG[pool]).all() and np.any(patch.BG[pool] != 0))
    best = None
    for k in range(len(pool)):
        order = np.argsort(distance[k])[:n_peers]             # ordered by seed distance
        if not np.isfinite(distance[k][order]).all():
            continue
        rows = pool[order]
        score, event_fraction, basis = _group_score(patch, rows, use_background)
        if not np.isfinite(score):
            continue
        if best is None or score < best["score"]:
            best = {"score": float(score), "seed_row": int(pool[k]), "rows": rows,
                    "distances": distance[k][order].astype(np.float32),
                    "event_fraction": event_fraction, "score_basis": basis}
    if best is None:
        raise RuntimeError("no candidate group passed the quiet-reference requirements")

    best["chip"] = chip
    if verbose:
        print(f"quiet reference context for chip {chip}: seed row {best['seed_row']}, score "
              f"{best['score']:.4f} ({best['score_basis']}), flagged-and-valid fraction "
              f"{best['event_fraction']:.4f}", flush=True)
    return best


def chip_suffix(chip):
    return f"_s{chip[0]:04d}_cam{chip[1]}_ccd{chip[2]}"


def save_reference_context(patch, reference, out_dir, primary=True):
    """reference_context.json (provenance) + reference_context.pt (curves/masks).

    `primary` also writes the unsuffixed filenames, so single-chip runs and older
    tooling keep working unchanged.
    """
    rows = reference["rows"]
    sector, camera, ccd = reference.get("chip", patch.chips[0])
    payload = {
        "peer_rows": torch.from_numpy(np.asarray(rows, np.int64)),
        "peer_raw": torch.from_numpy(patch.X[rows]),
        "peer_mask": torch.from_numpy(patch.M[rows]),
        "peer_flagged": torch.from_numpy(patch.Q[rows]),
        "peer_tess_flags": torch.from_numpy(patch.F[rows]),
        "peer_tglc_flags": torch.from_numpy(patch.G[rows]),
        "cadence_ids": torch.from_numpy(patch.grids[sector].copy()),
        "peer_detector_x": torch.from_numpy(patch.det_x[rows].astype(np.float32)),
        "peer_detector_y": torch.from_numpy(patch.det_y[rows].astype(np.float32)),
        "peer_distances": torch.from_numpy(reference["distances"]),
    }
    os.makedirs(out_dir, exist_ok=True)
    suffix = chip_suffix((sector, camera, ccd))
    torch.save(payload, os.path.join(out_dir, f"reference_context{suffix}.pt"))
    if primary:
        torch.save(payload, os.path.join(out_dir, "reference_context.pt"))

    meta = {
        "sector": sector, "camera": camera, "ccd": ccd,
        "split": "train",
        "seed_row": reference["seed_row"],
        "seed_tic": patch.tic[reference["seed_row"]],
        "seed_detector_x": float(patch.det_x[reference["seed_row"]]),
        "seed_detector_y": float(patch.det_y[reference["seed_row"]]),
        "peer_tics": [patch.tic[r] for r in rows],
        "peer_detector_x": [float(patch.det_x[r]) for r in rows],
        "peer_detector_y": [float(patch.det_y[r]) for r in rows],
        "peer_distances_from_seed": [float(d) for d in reference["distances"]],
        "selection_score": reference["score"],
        "score_basis": reference["score_basis"],
        "flagged_valid_fraction": reference["event_fraction"],
        "max_systematic_fraction": MAX_SYSTEMATIC_FRACTION,
        "cadence_id_first": int(patch.grids[sector][0]),
        "cadence_id_last": int(patch.grids[sector][-1]),
        "note": ("Observed low-systematics context used as a counterfactual baseline "
                 "condition. Not a correction target and not ground truth."),
    }
    with open(os.path.join(out_dir, f"reference_context{suffix}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    if primary:
        with open(os.path.join(out_dir, "reference_context.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
    return meta


def load_reference_context(out_dir, expected_cadence_ids=None, chip=None):
    name = "reference_context.pt" if chip is None else f"reference_context{chip_suffix(chip)}.pt"
    payload = torch.load(os.path.join(out_dir, name), weights_only=True)
    if expected_cadence_ids is not None:
        assert torch.equal(payload["cadence_ids"],
                           torch.as_tensor(expected_cadence_ids, dtype=payload["cadence_ids"].dtype)), \
            "quiet reference sits on a different absolute cadence grid than the target"
    return payload
