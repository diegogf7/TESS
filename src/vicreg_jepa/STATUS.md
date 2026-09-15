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
| Part 1 saved and frozen | **Yes** — best-validation checkpoint, bit-identity asserted through Part 2 |
| Pilot sweep completed | **No** — harness written and exercised, full 4-cell grid not run |
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

**The spec's λ=25 / μ=1 collapses on real data too.** Effective rank fell from
9.8 to 7.2 out of 64 over the first 100 steps, with median std 0.129 against
γ=1.0 — the pilot's own rejection rule flags it `COLLAPSED`. This reproduces the
synthetic finding. The sweep exists precisely to settle this; μ=25 was the
non-collapsing cell synthetically.

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
