# instrument_v2 — Sector-14 chip-signal experiments

## Files
- `diagnose_chip_common_signal.py` — Step-1/2 diagnostic: per-chip PCA common-mode
  classifier over three grid representations, with controls. Established that the
  chip signal is a strong, multi-dimensional, temporally shared common mode, best
  captured on a shared absolute-time grid.
- `sector14_dataset.py` — frozen splits + grids + `Sector14ChipPairDataset`.
- `train_sector14_jepa.py` — the JEPA experiment (this README).
- `eval_sector14_jepa.py` — frozen-encoder probe evaluation.

## JEPA experiment design
- **Data**: `tglc_raw_cadence_s14.parquet`; RAW `flux` (TGLC aperture_flux) only —
  `flux_cal`/`cal_aper_flux` are never read. Every finite raw-flux cadence is
  retained: quality-flag cleaning was a *diagnostic* condition only, and the
  Step-2 result showed a large part of the chip signal lives in flagged
  (systematics-heavy) cadences — removing them would delete the very signal the
  instrument encoder is supposed to learn.
- **Positive pairs**: two different TICs, same camera×CCD, Sector 14. Chip groups
  are sampled uniformly before stars (row-uniform sampling would let camera 1,
  ~66% of rows, dominate).
- **Grids (two arms, identical everything else)**:
  - `shared` (primary): one global Sector-14 time range → 1024 bins for every star.
  - `legacy` (control): per-curve local 1024 grid.
  - exact-cadence is NOT trained: the diagnostic showed it underperforms both.
- **Gaps — no infilling, mask passed through.** Unobserved bins hold normalized
  zero and the observed mask goes to the S4D encoder and to the masked loss.
  Justification: the mask-only control in the Step-1/2 diagnostics classified
  chips at **exactly chance (0.0625) within Sector 14**, so the mask channel
  carries no chip information here and gap-blind infilling (which is itself
  detectable, gap-audit AUC 0.814) is unnecessary for this single-sector
  experiment.
- **Model**: unchanged `GapBlindInstrumentJEPA` (S4D context encoder, EMA target
  encoder, latent predictor) with `gapblind_loss` (masked smooth-L1 + raw-token
  log-hinge spread penalty). No contrastive term.
- **Splits**: train/test TIC lists are the frozen Step-1 diagnostic artifacts;
  10% of train TICs become validation (seed 43, saved). Train/val/test are
  asserted mutually disjoint; test TICs never enter pretraining, checkpoint
  selection (checkpoints are every-10-epochs + final, no best-val), or probe
  hyperparameter tuning (validation only).

## Running
```
# train (2 arms x 3 seeds)
ARM=shared SEED=0 python -m src.instrument_v2.train_sector14_jepa
# evaluate all checkpoints + random baseline + step-1 PCA reference
python -m src.instrument_v2.eval_sector14_jepa
# tests
python -m src.tests.test_sector14_jepa
```
