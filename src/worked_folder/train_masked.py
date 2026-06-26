"""
Self-supervised pretraining of MaskedS4D on the real TESS curves (cluster, GPU).

Same data / paths / loop shape as train_physics_jepa.py, but the objective is
masked reconstruction instead of the old BYOL two-view matching. Only flux +
observed-mask are used -- no labels.

    python -m src.bot_folder.train_masked
"""

import os

import torch

from src.data.data import LightCurveDataset, DataLoader
from src.worked_folder.masked_s4d import build_masked_s4d, masked_recon_loss

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val_30min.parquet"
CHECKPOINT = os.environ.get("MASKED_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/masked_s4d.pth")
print(f"masked-S4D checkpoint -> {CHECKPOINT}")

dataset = LightCurveDataset(DATA_PATH, grid_length=1024)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = LightCurveDataset(VAL_PATH, grid_length=1024)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = build_masked_s4d().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (flux, mask) in enumerate(dataloader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        recon, seg_mask = model(flux, mask)
        loss = masked_recon_loss(recon, flux, seg_mask, model.patch, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {total_loss/len(dataloader):.6f}")
    scheduler.step()

    # validation
    model.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for flux, mask in val_dataloader:
            flux = flux.to(DEVICE)
            mask = mask.to(DEVICE)
            recon, seg_mask = model(flux, mask)
            val_total_loss += masked_recon_loss(recon, flux, seg_mask, model.patch, mask).item()

    avg_val = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, average val loss: {avg_val:.6f}")

    # keep the best-by-val checkpoint (recon loss is a clean early-stopping signal)
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  saved new best checkpoint (val {avg_val:.6f}) -> {CHECKPOINT}")
