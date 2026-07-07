"""Smoke test for the JEPA_PREDICTOR switch -- mirrors the real train call sites."""
import os

os.environ["JEPA_NTOKENS"] = "16"
os.environ["JEPA_READOUT"] = "mean_std"

import torch

from src.worked_folder.latent_jepa import build_latent_jepa, jepa_latent_loss


def n_params(module):
    return sum(p.numel() for p in module.parameters())


for kind in ["transformer", "mlp"]:
    os.environ["JEPA_PREDICTOR"] = kind
    model = build_latent_jepa()

    flux = torch.randn(4, 1024)
    pred, tgt, seg_mask = model(flux)
    loss = jepa_latent_loss(pred, tgt, seg_mask, var_weight=0.05)
    loss.backward()
    model.update_target()

    assert pred.shape == tgt.shape == (4, 16, 16), (kind, pred.shape, tgt.shape)
    expected = "MLPPredictor" if kind == "mlp" else "LatentPredictor"
    assert type(model.predictor).__name__ == expected, type(model.predictor).__name__

    print(f"{kind:12s} loss={loss.item():.4f}  predictor_params={n_params(model.predictor):,}")

print("smoke test passed")
