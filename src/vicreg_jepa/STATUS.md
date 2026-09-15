# Experiment status

Against the "definition of complete" in the completion instructions.
Last updated from the real-data pilot on the local workstation (CPU only).

## What has been done

| Requirement | Status |
|---|---|
| Real TESS data used | **Yes** — 14,984 real TGLC curves, Sector 1, cameras 1/2/4, 5 CCDs |
| Split manifest and data config saved | **Yes** — `curves_manifest.json`, with every TIC per split, the config, the npz SHA-256 and the git SHA |
| Corrected architecture preserved | **Yes** — all three deviations kept and regression-tested |
| Automated checks (spec §8) | **Yes** — 15/15 real-data checks pass, 21/21 synthetic |
| Part 1 trained and validated on real data | **Partial** — 400-step CPU pilot only |
| Part 2 trained on real data | **Partial** — 400-step CPU pilot; collapsed under spec weights |
| Part 1 saved and frozen | **Yes** — best-validation checkpoint, bit-identity asserted through Part 2 |
| Pilot sweep completed | **No** — harness written and exercised, full 4-cell grid not run |
| Collapse-gated search (≥100 trials) | **No** — harness written, refusal path verified on 4 trials |
| Ablations + 3-seed final runs | **No** — harness written, not run |
| Test set evaluated exactly once | **No** — deliberately not touched |
| Final table, mean ± sd across seeds | **No** |
| Training curves / collapse diagnostics | Logged per step to `metrics.jsonl`; no plots generated |

**The experiment is NOT complete.** Synthetic smoke tests do not count, and
neither does a 400-step CPU pilot.

## Why it is not complete

Two hard blockers on this workstation, neither of which code can remove.

**No GPU.** The protocol needs roughly 20 training runs (4 pilot cells + 5
ablations × 3 seeds). On CPU one 400-step small-model run takes ~25 minutes; the
real configuration is 8,000 Part 1 steps and 30,000 Part 2 steps at
`d_model=256, n_layers=4`. This needs the cluster — `submit_vicreg_jepa.sh`
stages all of it for `ou_mki_gpu`.

**No PhyTS physics labels here.** `physics_bacc` is the spec's primary metric
and every probe returns `NaN` without labels. The cache carries a `label` column
and `--label-csv` populates it; nothing else changes.

There is also a **data-scope gap**: the only real curves on this machine are
Sector 1. The existing baselines are Sector 14 / dense_v2. Comparisons in spec
§6.5–6.6 require rebuilding the cache on the same cohort those baselines used.

## Real-data findings so far

**Part 1 works and beats its baseline.** Best validation common-mode loss
**3.90** against a region-median baseline of **33.61** — 8.6× better. Predicting
a held-out star from the mean of its 31 region peers' latents is substantially
better than taking the peers' median flux, so the encoder is learning real
shared structure rather than copying.

**Part 1 is unstable at `lr=1e-3`.** Training spiked to 342.7 at step 300 (from
4.41 at step 200) and validation followed to 219.2. Gradient clipping at
`grad_clip=1.0` has been added to both loops; the spike predates it. Watch
`grad_norm` in `metrics.jsonl` on the first cluster run.

**The spec's λ=25 / μ=1 collapses on real data too.** Over 400 steps effective
rank fell 9.8 → 7.2 → 3.5 → 1.9 → **1.7 out of 64**. This reproduces the
synthetic finding. The sweep exists precisely to settle this; μ=25 was the
non-collapsing cell synthetically.

**The spec's collapse rejection rule is not sufficient, and it corrupted
selection.** At step 399 the latent had median std 0.667 and only 15% of
dimensions below 0.5 — clearing both of the spec's conditions — while its
effective rank was 1.7 of 64 and its off-diagonal covariance had exploded from
1.10 to 11.72 in a hundred steps. The latent was effectively one-dimensional and
the rule passed it.

The mechanism: `L_var` is a per-dimension hinge with no cross-dimension term, so
it can satisfy a std floor by inflating dimensions that all point the same way.
Std measures each axis alone; nothing in the spec's rule measures whether the
axes are distinct.

The damage was concrete. Steps 100–300 were flagged collapsed and skipped, so
the only checkpoint eligible for saving was step 399, whose `val_inv` of 0.0411
is *worse* than step 300's 0.0150. A rule meant to reject bad configurations
selected the worst available one.

`is_collapsed` now applies the spec's two conditions plus an effective-rank
floor at 25% of the latent width, and `collapse_reasons` records which fired.
Under the corrected rule step 399 is rejected for `effective_rank<16.0`.

## Collapse-gated search

The sweep was rebuilt so that a checkpoint failing any collapse check is never
saved, never ranked and never selected. See `README.md` for the gate and the
three-stage protocol.

**The ≥100-trial search has NOT been run.** Same three blockers: no GPU for the
~780,000 training steps the three stages require (100×2.5k + 12×10k×2 +
3×30k×3), no Sector 14 / dense_v2 cohort on this machine, and no PhyTS labels —
which are ranking criterion 1. Without labels the search runs but emits a
warning that the ranking is *not* the specified one, rather than silently
degrading to correlation-only selection.

**Machinery verified end-to-end on real curves.** A 4-trial / 60-step demo:

- the known-collapsing control (`phi=1, lambda=25, mu=1, nu=0.01`) was pruned
  after 2 consecutive rejected validations and recorded `eligible=False`
- all four trials failed the gate, and the search wrote
  `"selected": null, "status": "NO VALID CONFIGURATION FOUND"` with an expanded
  search space — it did not promote the least-collapsed trial

**This demo says nothing about the hyperparameter space.** Sixty steps is far
too few for any configuration to leave its initial collapsed state (latent
starts at effective rank ~8, median std ~0.10). It demonstrates that the refusal
path works, and nothing more.

## To finish on the cluster

```bash
export S14_DATA=...            # parquet glob for the target cohort
export LABEL_CSV=...           # PhyTS labels: TIC,label

sbatch --export=ALL,STAGE=cache      submit_vicreg_jepa.sh
sbatch --export=ALL,STAGE=part1      submit_vicreg_jepa.sh
sbatch --export=ALL,STAGE=pilot      submit_vicreg_jepa.sh
sbatch --export=ALL,STAGE=ablations  submit_vicreg_jepa.sh
# then, once and only once, with RUNS built from the ablation checkpoints:
sbatch --export=ALL,STAGE=final,RUNS='{"full_vicreg":["...pt"]}' submit_vicreg_jepa.sh
```
