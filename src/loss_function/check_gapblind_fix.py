"""Verify the collapse fix end to end on CPU, including a mini-training run
that demonstrates the variance penalty holds the raw latent spread open.

    python -m src.loss_function.check_gapblind_fix
"""

import torch

from src.loss_function.gap_infill import infill_gaps
from src.loss_function.gapblind_fix import GapBlindInstrumentJEPA, gapblind_loss
from src.worked_folder.instrument.instrument_jepa import InstrumentJEPA

torch.manual_seed(0)

B, L, N = 64, 256, 16
patch = L // N


def tiny(cls):
    return cls(n_tokens=N, token_dimension=16, d_model=32, n_layers=1, dropout=0.0)


# 1. checkpoint compatibility: gapblind state_dict loads into the plain class
sub = tiny(GapBlindInstrumentJEPA)
plain = tiny(InstrumentJEPA)
plain.load_state_dict(sub.state_dict())  # strict by default; raises on mismatch
print("state_dict compatible with plain InstrumentJEPA (probe will load it)")

# 2. synthetic batch: distinct sine curves + noise, one gap token per curve
t = torch.linspace(0, 6.28, L)
freqs = torch.rand(B, 1) * 20 + 2
flux = 0.02 * torch.sin(freqs * t) + 0.005 * torch.randn(B, L)
mask = torch.ones(B, L)
for b in range(B):
    g = int(torch.randint(0, N, (1,)))
    mask[b, g * patch: (g + 1) * patch] = 0.0
flux = flux * mask

# 3. loss contract: gap tokens still contribute nothing (var term off to isolate)
ctx, _ = infill_gaps(flux, mask)
tgt_in, _ = infill_gaps(flux, mask)
prediction, target, context_tokens = sub(ctx, None, tgt_in, None)
base = gapblind_loss(prediction, target, context_tokens, target_mask=mask, var_weight=0.0)
target_vandal = target.clone()
for b in range(B):
    g = int((mask[b].reshape(N, patch).sum(dim=1) == 0).nonzero()[0])
    target_vandal[b, g] += 100.0
after = gapblind_loss(prediction, target_vandal, context_tokens, target_mask=mask, var_weight=0.0)
assert torch.isclose(base, after), "gap token leaked into the gapblind loss"
print("masked-loss contract holds: pure-gap tokens contribute nothing")

# 4. the actual fix: mini-train two arms and compare raw cross-curve spread
def mini_train(var_weight, steps=600):
    torch.manual_seed(1)
    model = tiny(GapBlindInstrumentJEPA)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    model.train()
    for _ in range(steps):
        c, _ = infill_gaps(flux, mask)
        tg, _ = infill_gaps(flux, mask)
        pred, targ, ctx_tok = model(c, None, tg, None)
        loss = gapblind_loss(pred, targ, ctx_tok, target_mask=mask, var_weight=var_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
    model.eval()
    with torch.no_grad():
        online_std = ctx_tok.std(dim=0).mean().item()   # what the penalty acts on
        filled, _ = infill_gaps(flux, mask)
        z = model.encode(filled, None)                   # EMA target = what the probe sees
        ema_std = z.reshape(B, -1).std(dim=0).mean().item()
    return online_std, ema_std, loss.item()

on_off, ema_off, loss_off = mini_train(var_weight=0.0)
on_fix, ema_fix, loss_fix = mini_train(var_weight=0.1)
print(f"no fix : online std {on_off:.4f}, EMA std {ema_off:.4f}  (the collapse)")
print(f"log fix: online std {on_fix:.4f}, EMA std {ema_fix:.4f}")
assert torch.isfinite(torch.tensor([loss_off, loss_fix])).all(), "loss went non-finite"
assert on_fix > 0.5, f"fix failed: online spread only {on_fix:.4f}"
assert ema_fix > 5 * max(ema_off, 1e-6), "EMA spread should clearly recover vs the collapsed arm"

print("collapse fix verified: log-hinge variance penalty restores cross-curve spread")
