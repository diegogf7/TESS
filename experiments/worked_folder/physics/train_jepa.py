"""
Self-supervised pretraining of LatentJEPA on the real TESS curves (cluster, GPU).

Predicts in REPRESENTATION space (predict masked-segment latents from context),
not data space. Saves to its OWN checkpoint so it never overwrites masked_s4d.pth.

    python -m src.bot_folder.train_jepa
"""

import os

import torch

from src.data.data import LightCurveDataset, DataLoader
from src.worked_folder.physics.latent_jepa import build_latent_jepa, jepa_latent_loss

BATCH_SIZE = 256
EPOCHS = 30
VAR_WEIGHT = 0.05  # gentle anti-collapse safety net
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet")
VAL_PATH  = os.environ.get("JEPA_VAL",  "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val_30min.parquet")

CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_transformer.pth")
print(f"config: NTOKENS={os.environ.get('JEPA_NTOKENS','16')} READOUT={os.environ.get('JEPA_READOUT','mean')} "
      f"MASK_RATIO={os.environ.get('JEPA_MASK_RATIO','0.5')} -> {CHECKPOINT}")

dataset = LightCurveDataset(DATA_PATH, grid_length=1024)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_dataset = LightCurveDataset(VAL_PATH, grid_length=1024)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = build_latent_jepa().to(DEVICE)

# only the context encoder + predictor train; the EMA target is frozen
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_idx, (flux, mask) in enumerate(dataloader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        pred, tgt, seg = model(flux, mask)
        loss = jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT)
        loss.backward()
        optimizer.step()
        model.update_target()  # EMA step

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {total_loss/len(dataloader):.6f}")
    scheduler.step()

    # validation (collapse-aware: also report target-latent std)
    model.eval()
    val_total_loss = 0.0
    latent_chunks = []
    with torch.no_grad():
        for flux, mask in val_dataloader:
            flux = flux.to(DEVICE)
            mask = mask.to(DEVICE)
            pred, tgt, seg = model(flux, mask)
            val_total_loss += jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT).item()
            z = model.encode(flux, mask)
            latent_chunks.append(z.reshape(z.shape[0], -1).cpu())

    avg_val = val_total_loss / len(val_dataloader)
    latent_std = torch.cat(latent_chunks).std(0).mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS}, average val loss: {avg_val:.6f}, latent std (collapse check): {latent_std:.4f}")

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  saved new best checkpoint (val {avg_val:.6f}) -> {CHECKPOINT}")
