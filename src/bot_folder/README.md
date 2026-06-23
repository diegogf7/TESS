# bot_folder — masked-prediction S4D (replacing physics_jepa)

Goal: a self-supervised encoder whose frozen latent makes **SECTOR** decodable
with high accuracy (the result to show the professor). Self-supervised only,
staying inside the existing S4D + masking + predictor framework.

## 1. Why the old physics_jepa didn't work

`physics_jepa.py` is named I-JEPA but is really a BYOL-style two-view model, and
a degenerate one:

- **No real prediction target.** `forward` makes two *augmented views of the
  whole curve*, encodes each into 16 globally-pooled tokens, and trains an MLP
  to map view-A tokens → view-B tokens. The masking only zeroes inputs; it never
  defines a *hidden region to predict*. The task collapses to "make two
  augmentations match."
- **Collapse-prone loss.** `jepa_loss` L2-normalises each **4-dim** token then
  takes MSE. Normalising a 4-dim vector + EMA target + no variance regulariser
  is a classic recipe for representational collapse → probe sits at chance.
- **Augmentations can erase the physics** (amplitude scaling + heavy noise force
  invariance to exactly the things that distinguish some classes).

## 2. The fix (your idea: move the masking to the predictor, drop JEPA)

`masked_s4d.py` — a masked autoencoder for light curves:

1. patchify the curve into `n_tokens` contiguous segments,
2. randomly **hide** half the segments (zero their flux),
3. run the shared `S4Model` over the partially-blanked curve — the SSM
   convolution carries context into the hidden segments,
4. pool each segment to a token; a small MLP decoder reconstructs the **flux** of
   the hidden segments,
5. loss = MSE on hidden segments, observed points only.

The target is the real flux, not a moving network output, so **nothing can
collapse**. To fill the gaps the encoder must learn the curve's temporal
structure.

### Architecture finding you should know
`S4Model` **mean-pools each segment**, so any oscillation with a period shorter
than one segment is averaged away. The latent therefore captures segment-scale
and slower structure — which is exactly where instrument/sector systematics
live (good for the sector target), but means very short-period classes lose
detail. At `grid_length=1024, n_tokens=16` a segment is 64 points ≈ 1.3 days.

## 3. Empirical evidence (measured locally, `validate_synthetic.py`)

Real data lives on the cluster, so the method is proven on a **synthetic**
dataset with known ground truth: each curve carries a class oscillation and a
**low-amplitude sector systematic buried in noise**, with noise level and gap
positions identical across sectors (so sector can't be recovered by any
shortcut — only by extracting coherent structure). Run:

```bash
python -m src.bot_folder.validate_synthetic   # CPU, ~4 min
```

Measured (seed 0, 2000 curves, 6 sectors, 4 classes):

| metric | random-init encoder | after masked pretraining | chance |
|---|---|---|---|
| held-out variance explained | −0.81 | **0.22** | 0 |
| **SECTOR balanced accuracy (target)** | 0.35 | **0.78** | 0.17 |
| CLASS balanced accuracy | 0.72 | **0.94** | 0.25 |

Reading it: pretraining takes the model from worse-than-mean to explaining 22%
of held-out variance, and **sector becomes decodable at 78% balanced accuracy —
2.2× the no-pretraining baseline and ~4.7× chance.** The wide trained-vs-random
gap shows the *pretraining* creates the structure, not the architecture alone.
This is evidence for the *method*; the cluster run confirms the numbers on real
TESS data.

## 3b. Real-data result (cluster, job 16406515)

Trained 100 epochs on the real 30-min TESS curves, then probed the frozen latent:

| metric | value | chance |
|---|---|---|
| latent collapse check (mean std) | 0.152 | (want ≫ 0) ✅ |
| **SECTOR balanced accuracy (target)** | **0.601** | 0.056 (18 sectors) |
| CLASS balanced accuracy | 0.473 | 0.143 (7 classes) |

Sector is recovered at ~11× chance from a frozen, label-free latent — the
instrument signature is clearly captured. No collapse (the old physics_jepa's
failure mode). Class ≈ 2× the old JEPA's ~0.24, though below the ~0.88
supervised ceiling as expected for an SSL probe. Make the figures with
`umap_masked.py` (see below).

## 3c. LatentJEPA — predicting in REPRESENTATION space (not data space)

`latent_jepa.py` is the "work with JEPA in latent space" model. The predictor's
output is a **latent** and the loss is in latent space — the defining property of
JEPA. It is the fixed version of the old collapsing `physics_jepa`, via: real
masked-target prediction (predict each hidden segment's target latent at its
position), target LayerNorm (not the old 4-D L2), wider latent (token_dim 16),
and an optional variance safety net.

Validated locally (`validate_jepa_synthetic.py`): it trains (loss 0.29 → 0.03)
and — the key worry — **does NOT collapse** (latent std 0.10 → 0.17). Honest
caveat: on synthetic, its frozen probe ≈ a random encoder, because predicting
the encoder's *own* latent is self-referential and gives weaker probes than
reconstructing the real data (a known SSL property; matches the old JEPA_2 ~38%
stall). On real TESS (where random-init is weaker) it has a real shot at beating
random — that's what the cluster run tests.

Cluster run (separate checkpoint `latent_jepa.pth`, does NOT touch
`masked_s4d.pth`):
```bash
sbatch src/bot_folder/run_jepa.sh        # train_jepa -> eval_jepa, log latent_jepa_%j.out
```

## 4. Run order on the cluster

```bash
python -m src.bot_folder.train_masked   # GPU: self-supervised pretraining
python -m src.bot_folder.eval_masked    # GPU: freeze + KNN-probe SECTOR & CLASS
```

`build_masked_s4d()` in `masked_s4d.py` is the single source of architecture
truth shared by train and eval, so the checkpoint always loads. `eval_masked.py`
prints a collapse check (mean latent std) and the SECTOR/CLASS balanced
accuracies.

## 5. Files

| file | role |
|---|---|
| `masked_s4d.py` | **the model** + `build_masked_s4d()` + `masked_recon_loss` |
| `validate_synthetic.py` | **self-contained empirical proof** (CPU) |
| `train_masked.py` | cluster pretraining on real curves |
| `eval_masked.py` | cluster probe: SECTOR (target) + CLASS |
| `s4d.py`, `dropout.py`, `data.py`, `disentangle.py` | framework reference copies (the live code is imported from `src/`) |
| `physics_jepa.py`, `train_physics_jepa.py`, `umap_physics.py` | the superseded baseline, kept for comparison |

Note: the new code imports the shared encoder/datasets from `src/` (same
convention as the rest of the repo); the copies above are reference snapshots.
