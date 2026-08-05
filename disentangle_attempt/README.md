# disentangle_attempt — cross-sector physics / instrument split

An MVP that tries to separate a TESS star's own signal from its detector
neighbourhood's shared measurement behaviour, using **only raw TGLC aperture flux**.
No `cal_aper_flux`/`flux_cal`, no cadence alignment between sectors, no CBVs, no flow
matching, no attention/transformers, no covariance or correction-size losses, no
correlation-selected peers.

## The idea in one paragraph

One anchor star in a target sector is reconstructed from two latents. The **physics**
latent comes from the anchor's own curve with ~25% of its valid cadences hidden, so it
carries that star's timing (transits, flares, variability) but cannot copy the answer.
The **instrument** latent comes from eight *different* stars on the same
sector/camera/CCD and the same absolute cadence grid, picked purely by detector
distance — the branch never sees the anchor TIC. A shared MLP decoder gets both and
must predict the anchor's hidden cadences. A third view, the same TIC in a *different*
sector, only pulls the two sectors' pooled physics vectors together; it is a training
regularizer and is not needed at inference.

At inference the star's real neighbourhood is swapped for the quietest eight-peer group
actually observed on that chip. The result is a **counterfactual**: what the decoder
predicts under a quieter observed instrument condition. It is not proven ground truth
and not a measured correction.

## Files

| File | Role |
|---|---|
| `fetch_data.py` | builds the cross-sector patch from MAST (see below) |
| `dataset.py` | cadence grid, gridding, eligibility, TIC splits, detector peers; `python -m disentangle_attempt.dataset <parquet>` audits any parquet |
| `masking.py` | contiguous training masks + four complementary inference masks |
| `model.py` | shared physics S4D, shared instrument S4D, one shared MLP decoder |
| `losses.py` | masked smooth-L1 + cross-sector cosine consistency |
| `reference_context.py` | selects and saves the quiet observed peer group |
| `train.py` | fast run, branch-use tests, quiet-reference selection |
| `infer.py` | complementary-mask counterfactual + sparse-systematics test |
| `smoke_test.py` | shapes, gradient flow, tiny overfit |
| `config_fast.yaml` / `config_orcd.yaml` | MVP run / full cluster run |
| `submit_orcd.sh` | SLURM job for ORCD |

## The data problem, and how `fetch_data.py` solves it

The experiment needs the **same TIC in two sectors**, on a chip dense enough to supply
eight detector-nearest peers. No parquet in this repository provides that:

* `tglc_raw_cadence_s14_dense_v2_xy.parquet` is Sector 14 only — there is no second
  sector for any star, so the cross-sector branch cannot exist at all.
* `tglc_raw_cadence_all.parquet` covers sectors 1–26, but `src/tglc/download_primary.sh`
  reservoir-samples ~8 000 stars per sector independently across the whole sky, so the
  same TIC is expected to repeat across sectors only very rarely. This has not been
  measured on the cluster copy — check before assuming, with
  `python -m disentangle_attempt.dataset <parquet>`, which reports multi-sector TICs
  and eligible anchors per chip.

`fetch_data.py` builds a purpose-made patch instead, without scanning multi-GB indexes:

1. MAST's per-sector TGLC bulk script is sorted by camera/CCD then Gaia source id and
   serves HTTP range requests, so a binary search over byte offsets lands on the
   requested chip in ~25 4 KB reads.
2. Gaia DR3 source ids are HEALPix-ordered, so consecutive lines are a spatially
   compact patch — exactly the dense detector neighbourhood peer selection wants.
3. Camera 4 in sectors 1–13 watches the southern continuous viewing zone, so those
   stars are re-observed in most of those sectors.
4. `tess-point` maps each star's RA/Dec to `(sector, camera, ccd, column, row)` for
   every sector at once — it picks the partner sector *and* supplies the official
   `DETECTOR_X`/`DETECTOR_Y` (the same source of truth as
   `src/tglc/merge_detector_positions.py`; never the aperture-local `STAR_X`/`STAR_Y`).
5. A TGLC path is a pure function of the Gaia id, so the partner sector's URLs are
   constructed directly.

The default patch is Sector 1, camera 4, CCD 2 → partner Sector 12, 2 000 stars per
sector, ~290 MB, ~10 min.

```bash
python -m disentangle_attempt.fetch_data           # env: N_STARS, SECTOR_A, CAMERA, CCD, DA_DATA_DIR
```

## Data contract

* **Flux**: raw TGLC `aperture_flux`, median/MAD normalized with the repository's
  `normalize_median_mad`.
* **Cadence grid**: each sector gets one absolute grid — the 1024 consecutive TGLC
  cadence numbers carrying the most observations. Curves are placed by *exact cadence
  number*; nothing is resampled, interpolated or smoothed. Anchor and peers therefore
  share the same absolute cadences by construction, while the anchor and its
  other-sector curve sit on different grids and are never compared per cadence.
* **Validity**: `valid_mask` marks genuinely observed finite cadences. Quality flags do
  **not** remove cadences — momentum dumps and scattered light are the systematics the
  instrument branch is supposed to explain. The severe-flag mask
  (`BAD_TESS_MASK = 16437`) is carried alongside for evaluation and for the quiet
  reference ranking.
* **Splits**: TIC-keyed 80/10/10. Peers come only from the anchor's own split, so a TIC
  cannot leak across splits through the peer or the cross-sector branch.
* **Eligibility**: same TIC in another sector, ≥ 8 other TICs on the chip, and at least
  `min_valid_fraction` of cadences valid.

## Architecture

```
masked anchor  [32,1024] ─ physics S4D ──→ [32,32,16] ─flatten→ 512 ┐
other sector   [32,1024] ─ physics S4D ──→ [32,32,16] ─mean,L2→ 16  │ (cosine only)
8 peers      [32,8,1024] ─ instrument S4D → [32,8,32,16] ──────→ 4096┤
                                                                     └→ MLP(4608→1024→1024) → [32,1024]
```

Both encoders are the repository's `S4Model` (`src/models/s4d.py`) with `n_tokens=32`,
`token_dim=16` hard-set — not inherited from the `instrument_v2` default of 16 tokens.
`S4Model` takes `(B, L, d_input)` and its masked token pooling drops missing cadences,
so a curve enters as `raw.unsqueeze(-1)`: one channel of length 1024, with the channel
axis where this S4D wants it. Peers are encoded by one shared instrument S4D in a
single flattened call and stay ordered nearest-to-farthest — never averaged. The
decoder is one MLP for every anchor; the existing `LatentCBVDecoder` could not be
reused because it predicts coefficients against a per-chip CBV basis rather than a
curve.

Loss, averaged over all 32 anchors with exactly one `backward()` per step:

```
loss = masked_smooth_l1(prediction, anchor_raw, hidden & valid) + 0.05 * (1 - cos(global_physics, other_sector_global_physics))
```

## Running

```bash
python -m disentangle_attempt.smoke_test
python -m disentangle_attempt.train --config disentangle_attempt/config_fast.yaml
python -m disentangle_attempt.infer \
  --checkpoint disentangle_attempt/outputs/<run_name>/best.pt --tic-id <TIC_ID>
```

On ORCD: `sbatch disentangle_attempt/submit_orcd.sh` (fetch the patch on the login
node first — compute nodes have no outbound network; see the script header).

## Mandatory tests

* **Smoke** — every tensor shape above, one forward/backward, and finite nonzero
  gradients in the physics S4D, the instrument S4D and the decoder.
* **Tiny overfit** — repeated training on 3 steps must drop masked reconstruction
  substantially.
* **Branch use** (`branch_use_tests.json`) — validation loss is re-measured with (1)
  physics inputs shuffled across TICs, (2) nearest peers replaced by random
  same-chip peers, (3) the cross-sector curve replaced by a different TIC. The loss
  geometry (target, validity, hidden mask) is held identical in every condition, so
  only the branch contents change. Physics shuffling must clearly worsen
  reconstruction; peer replacement should worsen it, and if it does not, the decoder is
  ignoring the instrument branch.
* **Sparse systematics** (`sparse_systematics.json`) — a smooth time-localized
  excursion is injected into 4 of 32 anchors *and their peers*, plus a transit into the
  anchors only. The actual-context reconstruction should follow the systematic, the
  quiet-context decode should suppress it, and the transit should survive cleaning.

## Results

See `outputs/<run_name>/metrics.json`. `RESULTS.md` records the local MVP run.

## What this does not prove

Disentanglement is *encouraged* by the branch inputs, not guaranteed by them. The
branch-use tests are what decide whether the model actually uses the division, and the
cleaned curve remains a quiet-condition counterfactual — not ground truth, and not a
measured correction. Do not expand the model until the branch-use tests pass.
