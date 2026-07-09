import torch

from src.worked_folder.instrument.instrument_jepa import instrument_jepa_loss

torch.manual_seed(0)

B = 4
N = 16
D = 16
L = 1024

patch = L // N

prediction = torch.randn(B, N, D)
target = torch.randn(B, N, D)

mask = torch.ones(B, L)
mask[:, 7*patch: 8*patch] = 0.0

baseline = instrument_jepa_loss(prediction, target, target_mask = mask)

target_gap = target.clone()
target_gap[:, 7] += 100.0

assert torch.isclose(baseline, instrument_jepa_loss(prediction, target_gap, target_mask = mask)), "gap leaked in loss"


#now for the second case
target_observed = target.clone()

target_observed[:, 3] += 100.0

assert not torch.isclose(baseline, instrument_jepa_loss(prediction, target_observed, target_mask = mask)), "observed token ignored"

print("gap tokens contribute nothing and observed tokens will count")