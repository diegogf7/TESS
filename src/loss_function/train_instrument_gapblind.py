"""Gap-blind retrain of the cross-star instrument JEPA.

SAME model as always -- no architecture changes. Two differences from the
gapmask run: (1) inputs are infilled (gaps replaced with values resampled
from their own observed points) and the encoders get no mask, so gap
placement is invisible; the ORIGINAL observed mask still weights the loss.
(2) the variance penalty sits on the encoder's RAW context tokens (see
gapblind_fix.py) -- the first gap-blind run collapsed to raw latent std
~0.006 because nothing held the raw spread open once the masks were hidden.

    JEPA_VARW=0.5 python -m src.loss_function.train_instrument_gapblind
"""

import os

import torch

from src.data.data import DisentanglementDataset, DataLoader
from src.loss_function.gap_infill import infill_gaps
from src.loss_function.gapblind_fix import build_gapblind_jepa, gapblind_loss

BATCH_SIZE = 256
EPOCHS = int(os.environ.get("JEPA_EPOCHS", "100"))
VAR_WEIGHT = float(os.environ.get("JEPA_VARW", "0.5"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet")
VAL_PATH = os.environ.get("JEPA_VAL", "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val_30min.parquet")
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_gapblind.pth")

dataset = DisentanglementDataset(DATA_PATH, grid_length=1024)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = DisentanglementDataset(VAL_PATH, grid_length=1024)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"config: EPOCHS={EPOCHS} VAR_WEIGHT={VAR_WEIGHT} -> {CHECKPOINT}")
model = build_gapblind_jepa().to(DEVICE)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (anchor_flux, anchor_mask, ss_flux, ss_mask, same_sector_flux, same_sector_mask) in enumerate(dataloader):
        anchor_flux = anchor_flux.to(DEVICE)
        anchor_mask = anchor_mask.to(DEVICE)
        same_sector_flux = same_sector_flux.to(DEVICE)
        same_sector_mask = same_sector_mask.to(DEVICE)

        # hide the gaps from both encoders; keep the real mask for the loss
        context_flux, _ = infill_gaps(same_sector_flux, same_sector_mask)
        target_flux, _ = infill_gaps(anchor_flux, anchor_mask)

        optimizer.zero_grad()
        prediction, target, context_tokens = model(context_flux, None, target_flux, None)
        loss = gapblind_loss(prediction, target, context_tokens, target_mask=anchor_mask, var_weight=VAR_WEIGHT)
        loss.backward()
        optimizer.step()
        model.update_target()  # EMA step

        total_loss = total_loss + loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1} / {EPOCHS}, batch {batch_idx}, loss {loss.item(): .5f}")

    print(f"Epoch {epoch + 1} / {EPOCHS}, average loss: {total_loss / len(dataloader): .5f}")
    scheduler.step()

    model.eval()
    val_total_loss = 0.0
    latent_chunks = []

    with torch.no_grad():
        for anchor_flux, anchor_mask, ss_flux, ss_mask, same_sector_flux, same_sector_mask in val_dataloader:
            anchor_flux = anchor_flux.to(DEVICE)
            anchor_mask = anchor_mask.to(DEVICE)
            same_sector_flux = same_sector_flux.to(DEVICE)
            same_sector_mask = same_sector_mask.to(DEVICE)

            context_flux, _ = infill_gaps(same_sector_flux, same_sector_mask)
            target_flux, _ = infill_gaps(anchor_flux, anchor_mask)

            prediction, target, context_tokens = model(context_flux, None, target_flux, None)
            val_total_loss = val_total_loss + gapblind_loss(prediction, target, context_tokens, target_mask=anchor_mask, var_weight=VAR_WEIGHT).item()

            z = model.encode(target_flux, None)
            latent_chunks.append(z.reshape(z.shape[0], -1).cpu())

    average_val = val_total_loss / len(val_dataloader)
    latent_std = torch.cat(latent_chunks).std(0).mean().item()
    print(f"Epoch {epoch + 1} / {EPOCHS}, average val loss: {average_val:.5f}, latent std (collapse checking): {latent_std: .5f}")

    if average_val < best_val:
        best_val = average_val
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"saved new best checkpoint (val {average_val:.6f}) --> {CHECKPOINT}")
