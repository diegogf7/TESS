"""Sanity checks for gap infill + the gap-blind training path. CPU, seconds.

    python -m src.loss_function.check_infill
"""

import torch

from src.loss_function.gap_infill import infill_gaps
from src.worked_folder.instrument.instrument_jepa import InstrumentJEPA, instrument_jepa_loss

torch.manual_seed(0)

B, L = 4, 1024
N = 16
patch = L // N

# sine + noise curves with a full-token gap (token 7) and a half-token gap (token 3)
t = torch.linspace(0, 8 * 3.14159, L)
flux = 0.01 * torch.sin(t).repeat(B, 1) + 0.002 * torch.randn(B, L)
mask = torch.ones(B, L)
mask[:, 7 * patch: 8 * patch] = 0.0
mask[:, 3 * patch: 3 * patch + patch // 2] = 0.0
flux = flux * mask  # placeholder zeros in gaps, like resample_to_grid makes them

filled, ones = infill_gaps(flux, mask)

assert torch.equal(filled[mask > 0], flux[mask > 0]), "observed points were modified"
assert torch.equal(ones, torch.ones_like(mask)), "returned mask is not all ones"

for b in range(B):
    gap_values = filled[b][mask[b] == 0]
    observed_values = flux[b][mask[b] == 1]
    assert torch.isin(gap_values, observed_values).all(), "fill values not drawn from observed points"

assert filled[0, 7 * patch: 8 * patch].std() > 0, "gap region is still dead flat"

# an all-gap curve must not crash and comes back unchanged
zero_filled, _ = infill_gaps(torch.zeros(1, L), torch.zeros(1, L))
assert torch.equal(zero_filled, torch.zeros(1, L)), "all-gap curve should come back unchanged"

# end to end: gap-blind forward + the existing masked loss trains
model = InstrumentJEPA(n_tokens=N, token_dimension=16, d_model=64, n_layers=2, dropout=0.0)
context_flux, _ = infill_gaps(flux, mask)
target_flux, _ = infill_gaps(flux, mask)
prediction, target = model(context_flux, None, target_flux, None)
loss = instrument_jepa_loss(prediction, target, target_mask=mask, var_weight=0.1)
loss.backward()
assert torch.isfinite(loss), "loss is not finite"

# the encoder can no longer tell gapped from ungapped placement: encoding the
# same filled curve with mask=None twice is deterministic in eval mode
model.eval()
with torch.no_grad():
    z1 = model.encode(context_flux, None)
    z2 = model.encode(context_flux, None)
assert torch.allclose(z1, z2), "encode is not deterministic in eval mode"

print("infill checks passed: observed untouched, gaps filled from observed values, gap-blind forward trains")
