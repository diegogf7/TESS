# Local MVP run — `outputs/fast_local`

Apple M-series GPU (MPS), 17.2 min, 8 epochs × 100 steps, stopped by `max_epochs`.
Mixed precision off (S4D needs complex FFT; half-complex is CUDA-only).

## Setup

| | |
|---|---|
| Data | Sector 1 cam4-ccd2, partner Sector 12; 1 986 TICs in both sectors |
| Anchors | 800 train / 100 val / 100 test (capped at `max_eligible_anchors: 1000`) |
| Peer pools | 1 589 / 199 / 198 curves, split-disjoint by TIC |
| Peer distance | median 6.4 detector px to the 8 nearest different TICs |
| Parameters | 6.18 M — physics S4D 202 k, instrument S4D 202 k, decoder 5.77 M |

## Training

Best epoch 7. Validation masked smooth-L1 0.1818 → **0.1187**; test 0.1511.
Cross-sector consistency loss 0.0196 (val).

Validation loss on *visible* cadences (0.1187) matches the hidden-cadence loss
(0.1187), so the decoder is not winning by copying — it reconstructs held-out and
seen cadences equally well.

## Branch-use tests — all three pass

| Control | reconstruction | Δ | verdict |
|---|---|---|---|
| baseline | 0.1187 | — | |
| physics inputs shuffled across TICs | 0.4854 | **+0.3666** | physics branch used |
| nearest peers → random same-chip peers | 0.2039 | **+0.0852** | instrument branch used |
| cross-sector curve → different TIC | 0.1187 | +0.0000 | (consistency **+0.0089**) |

The instrument branch is genuinely load-bearing: swapping the eight detector-nearest
peers for random peers on the same sector/camera/CCD costs **+72 %** reconstruction
error, even though both sets share the chip and the cadence grid. Only proximity
distinguishes them. Wrong cross-sector TICs leave reconstruction untouched (correct —
that branch never reaches the decoder) and worsen the global consistency metric.

## Sparse-systematics test — 2 of 3

A Gaussian excursion (4 MAD, ~40 cadences) was injected into 4 of 32 test anchors *and
their peers*, plus a 1.5-MAD transit into the anchors only.

| Check | Value | Pass |
|---|---|---|
| actual-context reconstruction follows the systematic | 0.385 of injected amplitude | yes |
| quiet-context decoding suppresses it | 0.029 (13× suppression) | yes |
| transit survives cleaning | 0.068 of injected depth | **no** |

The first two are the point of the experiment and they hold clearly — see
`sparse_systematics.png`, where the actual-context prediction tracks the injected bump
and the quiet-context one ignores it. Uninjected anchors show no response, so the
effect is local, not a global shift.

The transit check fails, and the honest reading is that this model cannot yet express a
sharp 26-cadence feature: the physics latent is 32 tokens pooled over 32 cadences each,
so one transit is roughly one token, and 8 epochs over 800 anchors is not enough to
learn that mapping. This is a capacity/training-scale limit, not evidence that cleaning
eats astrophysical signal — note the measurement is already taken on a pass where the
transit is *visible* to the encoder, because under complementary masking a hidden
transit is unrecoverable by construction. Re-check it after the ORCD run.

## Quiet reference context

Seed row 660, background-variability score 6.40, severe-flag fraction 0.0093, chosen
from the train split over all 1 589 candidate seed locations.

## Counterfactual plot

`example_correction.png` (TIC 261236568, test split). Correction RMS 0.4465 MAD. Two
features stand out and both are recognisable TESS instrument behaviour rather than
stellar signal: a ramp over the first ~80 cadences of each orbit (scattered light after
each downlink) and regular spikes about every 120 cadences (momentum dumps).

**The cleaned curve is a quiet-condition counterfactual — what the decoder predicts if
this star's detector neighbourhood had been as quiet as the quietest group actually
observed on this chip. It is not proven ground truth and not a measured correction.**

## Next

Scale on ORCD (`config_orcd.yaml`, `submit_orcd.sh`): all 1 986 anchors, `d_model` 256,
300 steps/epoch, up to 60 epochs, AMP actually enabled. Do not expand the model itself
until the transit check also passes.
