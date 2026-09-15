# VICReg-JEPA — two-part disentangling model for TESS light curves

Part 1 learns what a sky region **shares** (instrument systematics). Part 2
learns a physics latent by predicting a masked view of a curve from itself,
while being pushed away from that systematics latent.

## Deviations from the original specification

Three, all deliberate and all covered by regression tests.

**1. Part 1 is leave-one-out peer reconstruction, not pairwise decorrelation.**
The specified `L_decor = Σ|corr(z_i, z_j)|` has a trivial solution: 32 vectors
in 32 dimensions can be made exactly orthogonal without reading the input, so
`L_decor < 0.05` is reachable by an encoder that ignores the light curves. It is
also backwards — region peers *share* systematics, so a systematics encoder
wants the common mode, not decorrelation.

**2. `L_var`, `L_cov` and `L_cor_sys` attach to `z_masked`, not `z_unmasked`.**
`z_unmasked` comes from the EMA encoder under `torch.no_grad()`. Measured on a
real batch, the encoder gradient with the three terms attached there is
*bit-identical* to using `L_inv` alone — they are inert, leaving predictor +
detached-target MSE, which collapses. They also cannot sit on `z_pred`: the
predictor is a free MLP with enough capacity to satisfy them by itself while the
encoder collapses behind it. `z_masked` is the tensor that gets frozen for
downstream probes, so that is where the constraints belong.

**3. Encoder outputs are not batch-normalised.** That would pin std to 1 and
make `L_var` identically zero.

## Pipeline

| Stage | Command |
|---|---|
| Build cache | `python -m src.vicreg_jepa.real_data --parquet-glob '<glob>' --out artifacts/vicreg_jepa/curves.npz` |
| Checks (synthetic) | `python -m src.vicreg_jepa.test_pipeline` |
| Checks (real) | `python -m src.vicreg_jepa.test_real_data` |
| Part 1 | `python -m src.vicreg_jepa.train --part 1 --source real` |
| Pilot sweep | `python -m src.vicreg_jepa.sweep --stage pilot --part1-ckpt <ckpt>` |
| Search (≥100 trials) | `python -m src.vicreg_jepa.search --part1-ckpt <ckpt> --n-trials 100` |
| Ablations | `python -m src.vicreg_jepa.sweep --stage ablations --part1-ckpt <ckpt> --selection <search>/selection.json --seeds 0 1 2` |
| Final test | `python -m src.vicreg_jepa.final_eval --part1-ckpt <ckpt> --runs '<json>'` |

On the cluster, `submit_vicreg_jepa.sh` wraps each stage:
`sbatch --export=ALL,STAGE=part1 submit_vicreg_jepa.sh`.

## Data contract

Matches `src/instrument_v2/sector14_dataset.py` so results stay comparable with
the existing baselines. `test_real_data.py` asserts the local helpers are
bit-identical to the repo reference implementations.

- RAW `flux` only; `flux_cal` is never read.
- A cadence is dropped when `(TESS_flags & 16437) != 0`, `TGLC_flags != 0`, or
  the flux is not finite. Dropped cadences stay **missing** — never infilled.
- Median/MAD normalisation from surviving cadences only.
- Shared per-sector grid: one global time range per sector, 1024 equal bins,
  identical for every star. Unobserved bins hold `0.0` with mask `0`.
- Region id = `camera*100 + ccd*10 + ring`, ring 1..4 by angular distance from
  the camera boresight (`src/regions/areas.py` convention, needs `tess-point`).
- Splits are **TIC-disjoint**: every observation of a star lands in exactly one
  of train / val / test. The manifest records every TIC per split.

Masking hides 30–50% of each curve's **currently observed** cadences, not 30–50%
of the 1024 slots — on real curves only 79–95% of slots carry data, so sampling
over all slots would hide a coverage-dependent share of the real signal and make
the task easier for well-covered stars.

## Collapse gate

Standard deviation alone is never treated as proof that collapse was avoided. A
real pilot produced a latent with median std **0.667** whose effective rank was
**1.7 of 64**: the variance hinge had been satisfied by copying essentially the
same signal into every dimension. Every threshold below exists because std
missed something.

Three representations are measured separately at each validation:

| lane | what it is |
|---|---|
| `masked_online` | `encoder(masked view)` — what the VICReg terms act on |
| `full_online` | `encoder(full curve)` — diagnostic |
| `full_ema` | `ema_encoder(full curve)` — **the primary representation** |

A checkpoint is rejected when `full_ema` fails **any** condition, or when an
online lane fails on `online_persist` consecutive validations:

- effective rank < 16 of 64
- median std < 0.5
- more than 20% of dimensions below 0.5 std
- off-diagonal covariance ratio > 0.5 (scale-free), or an absolute explosion
- any non-finite value
- more than 5% of dimension pairs with |r| > 0.95 (nearly identical dimensions)

The redundancy gate is deliberately **scale-free** — off-diagonal covariance is
judged relative to the mean diagonal — so a latent that is merely larger in
scale is not mistaken for a collapsed one.

The **target** effective rank for a final model is 32 of 64. It is reported
separately from the gate and is never silently lowered: a selected model that
misses it is flagged with an unmet-objective warning and an expanded search
space.

## Search

Three stages, all selecting on validation only:

1. ≥100 trials × 2,000–3,000 steps. A trial is pruned after 2 consecutive
   rejected validations.
2. The best 10–15 non-collapsed configurations × ~10,000 steps × 2 seeds.
3. The best 3 × 30,000 steps × seeds 0, 1, 2.

Rules the code enforces rather than assumes:

- A trial is eligible only if **every** seed produced a checkpoint that passed
  every collapse check. Checkpoints are written only when the gate passes, so a
  collapsed state cannot be selected even by accident.
- A stage-3 configuration must additionally be collapse-free **throughout**
  training after warmup, on every seed. Late collapse rejects the configuration.
- There is no "least collapsed" fallback. If nothing is eligible the search
  reports `NO VALID CONFIGURATION FOUND` and proposes an expanded space —
  thresholds are never relaxed.
- `phi=1, lambda=25, mu=1, nu=0.01` is carried as an explicit control so its
  failure is recorded rather than assumed.
- Sampling is biased toward larger `nu` (about 70% of draws have `nu >= 1`),
  because `nu=0.01` proved too weak to stop dimensional redundancy.
- Ranking is lexicographic: PhyTS balanced accuracy, then within 0.01 lower
  physics–systematics correlation, then lower camera/CCD/area leakage, then
  higher effective rank and lower off-diagonal covariance, then invariance loss
  as the final tie-break. Never correlation or invariance alone.
- The selected hyperparameters propagate into the ablations via
  `--selection <search>/selection.json`.
- `search.py` deletes the test split from its sources before running.

## Protocol guardrails

- The pilot selects on **validation only**; `pilot.json` records the rule.
- A configuration is rejected as collapsed if median std < 0.5 or more than 20%
  of dimensions have std < 0.5.
- The primary representation is the **EMA encoder on the full observed curve**.
  The online encoder is reported as a diagnostic, never separately selected.
- `final_eval.py` writes `TEST_EVALUATED.json` and refuses to run twice, so the
  test set is scored exactly once.
- Part 1 parameters are fingerprinted before Part 2 and asserted bit-identical
  afterwards.

## Status

See `STATUS.md` for what has and has not been run.
