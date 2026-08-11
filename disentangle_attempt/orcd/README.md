# Scaling the two-decoder spread experiment to 80 chips, on ORCD

This is the completed Sector-1 / camera-4 / CCD-2 experiment
(`local_s1_c4_ccd2_twodecoder_p64_i4_spread`, test masked Smooth-L1 **0.13184**) run
over **5 sectors x 4 cameras x 4 CCDs = 80 chips**.

The model, the flux contract (TGLC `aperture_flux`), the masking, the loss and the
peer rule are unchanged. Only the amount of data changes.

---

## Run order

```bash
cd /orcd/scratch/orcd/006/diegogon/TESS

bash disentangle_attempt/orcd/00_preflight.sh            # login node, ~1 min

tmux new -s acquire                                      # login node, ~20-40 min
bash disentangle_attempt/orcd/01_acquire.sh

sbatch --partition=pg_mki_aryeh disentangle_attempt/orcd/02_extract.sbatch
sbatch --partition=pg_mki_aryeh disentangle_attempt/orcd/03_prepare.sbatch
sbatch --partition=ou_mki_gpu   disentangle_attempt/orcd/04_train.sbatch
```

Edit `disentangle_attempt/orcd/env.sh` to change paths, sectors or partitions.
Nothing else needs editing.

**Stage 1 must run on a login node.** It is the only stage that needs outbound
network, and compute nodes generally have none. Everything is resumable — re-running
any stage skips completed work.

---

## What each stage does

| Stage | Where | Time | Output |
|---|---|---|---|
| `00_preflight` | login | 1 min | verifies packages, network, disk |
| `01_acquire` | login | 20–40 min | ~14 GB FITS, 240k light curves |
| `02_extract` | CPU job | 1–3 h | per-chip parquet + detector positions |
| `03_prepare` | CPU job | 20–60 min | dataset cache + geometry audit |
| `04_train` | GPU job | hours | checkpoints, metrics, per-chip test scores |

Disk: **~22 GB** total under `$DATA_DIR` and `$OUT_DIR`.

---

## Two things worth knowing

**1. Why not `fetch_data.py`.** It takes a run of *consecutive* bulk-script lines.
Gaia ids are HEALPix-ordered, so that is a spatially compact patch — the existing
single-chip parquet spans only `X [1724, 2091]`, `Y [3, 222]`, a 367x219 px corner of
a 2048x2048 CCD. The outer `[384, 768]` px peer band is geometrically impossible in
it. `multichip_acquire` indexes the *complete* per-chip block and samples it
uniformly, giving full-detector coverage (measured: `X [44, 2091]`, `Y [0, 2047]`,
max separation ~2890 px on every pilot chip).

**2. Why the download uses `archive.stsci.edu` and not the MAST API.** The API
gateway rate-limits hard — 2525 of 3000 requests returned HTTP 429 at 64 concurrent
workers. The static archive path served the identical files at **249 files/s with
zero errors** at 32–128 workers. The gateway is kept only as a per-file fallback.

---

## Verification already done

- **The peer rule is provably unchanged.** `spread_geometry.py` is a port of the
  selector out of `spread_peers.py` (left untouched, since it carries the completed
  run's frozen manifest). `test_spread_geometry_port.py` replays the port over the
  completed run's own anchors and pools and requires identical peer TICs in identical
  slot order: **400 anchors compared, 0 mismatches**.

- **The geometry survives at 3000 stars/chip.** On a 5-chip pilot, the strict 256 px
  minimum peer separation held for **1953 of 2000 anchors (97.7%)**; 42 fell back to
  192 px, 5 to 128 px, 2 found no group. The single-chip run was 382/400 (95.5%) at
  strict 256 px, so the scaled data is no worse.

- **Per-chip anchor counts reproduce the original proportions.** The SHA-256 80/10/10
  TIC split over 400 candidates per chip gives ~320/40/40 per chip — the original
  experiment's exact split, once per chip. Over 80 chips: ~25,600 / 3,200 / 3,200.

- **The full pipeline ran end to end locally** on 5 chips, including the architecture
  check (which still asserts the exact 1,777,764 parameters, unchanged), the
  tiny-overfit gate, training, per-chip test metrics and held-out plots.

---

## What differs from the single-chip run, and why

| | Single chip | 80 chips |
|---|---|---|
| Anchors | 320 / 40 / 40 fixed | ~25,600 / 3,200 / 3,200 |
| TIC split | frozen legacy numpy RNG | SHA-256 80/10/10 hash |
| Epoch | 40 steps | 800 steps (a pass would be ~80x longer) |
| Validation | full 40 anchors | capped 1024/epoch, full split scored once at the end |
| Curve storage | full flag arrays in RAM | `X` float32 + `M` bool only, loaded chip by chip |

The split changes because a frozen single-chip manifest cannot cover 80 chips. The
hash split is the one already documented in `spread_peers.stable_peer_split`, and it
is applied to *every* TIC, so a star observed in several sectors lands in one split
globally and cannot leak between splits through a second sector.

The memory change is a hard requirement, not an optimization: `CrossSectorPatch` keeps
int64 `TESS_flags`/`TGLC_flags` grids and reads the whole parquet at once. At 240k
stars that is ~15 GB for the frame plus ~6.4 GB gridded, which would be OOM-killed.
Flag policy is instead asserted per chip at load time and then summarized.

---

## Interpreting the result

`metrics.json` carries the same warning the single-chip run did, and it still applies:

> These are learned additive components under asymmetric information routing.
> Reconstruction alone does not physically identify the decomposition; injection and
> peer-control tests remain future work.

The scale-up gives more data and a per-chip breakdown of test loss. It does **not**
by itself demonstrate that `physics_curve` and `instrument_correction` are physically
separated. `per_chip_test_metrics.csv` is the new diagnostic worth reading: if the
instrument branch is learning genuine detector systematics rather than memorizing one
chip, its behaviour should be broadly consistent across the 80 chips.
