import torch 
import torch.nn as nn

from src.data.data import SectorDataset, DataLoader
from src.models.physics_jepa import PhysicsJEPA, jepa_loss

BATCH_SIZE = 256
EPOCHS = 100
LAMBDA = 1.0                       # adversary strength (the main knob)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"

dataset = SectorDataset(DATA_PATH, grid_length=1024)
val_dataset = SectorDataset(VAL_PATH, grid_length=1024)


unique_sectors = sorted(int(s) for s in dataset.df["sector"].unique())
sector_to_idx = {s: i for i, s in enumerate(unique_sectors)}
n_sectors = len(unique_sectors)
print(f"{n_sectors} sectors")

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
val_dataloader= DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

model = PhysicsJEPA(d_model = 256, n_layers = 4, dropout = 0.2, momentum = 0.996, n_sectors = n_sectors, lambd = LAMBDA).to(DEVICE)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)
sector_criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (flux, mask, sector) in enumerate(dataloader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        sector_idx = torch.tensor([sector_to_idx[int(s)] for s in sector], device = DEVICE)

        optimizer.zero_grad()

        prediction, z_target, sector_logits = model(flux, mask)

        loss_jepa = jepa_loss(prediction, z_target)

        loss_adversial = sector_criterion(sector_logits, sector_idx)

        loss = loss_jepa + loss_adversial

        loss.backward()
        optimizer.step()

        model.update_target()
        total_loss = total_loss + loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, " f"jepa: {loss_jepa.item():.4f}, adv: {loss_adversial.item():.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS}, avg loss: {total_loss/len(dataloader):.6f}")
    scheduler.step()

    # validation: just track the jepa loss
    model.eval()
    val_total = 0
    with torch.no_grad():
        for flux, mask, sector in val_dataloader:
            flux = flux.to(DEVICE); mask = mask.to(DEVICE)
            prediction, z_target, _ = model(flux, mask)
            val_total += jepa_loss(prediction, z_target).item()
    print(f"Epoch {epoch+1}/{EPOCHS}, avg val jepa loss: {val_total/len(val_dataloader):.6f}")

    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/training_physics_dann.pth')


