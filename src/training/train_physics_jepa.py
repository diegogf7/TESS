import torch
import torch.nn as nn

from src.data.data import LightCurveDataset, DataLoader
from src.models.physics_jepa import PhysicsJEPA, jepa_loss

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val_30min.parquet"


dataset = LightCurveDataset(DATA_PATH, grid_length = 1024)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

val_dataset = LightCurveDataset(VAL_PATH, grid_length=1024)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model = PhysicsJEPA(d_model = 256, n_layers = 4, dropout = 0.2, momentum = 0.996).to(DEVICE)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for batch_idx, (flux, mask) in enumerate(dataloader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        prediction, z_target = model(flux, mask)
        loss = jepa_loss(prediction, z_target)

        loss.backward()
        optimizer.step()
        model.update_target() #for the EMA

        total_loss = total_loss + loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {total_loss/len(dataloader):.6f}")
    scheduler.step()


    #validation code for me now
    model.eval()

    val_total_loss = 0

    with torch.no_grad():

        for batch_idx, (flux, mask) in enumerate(val_dataloader):
            flux = flux.to(DEVICE)
            mask = mask.to(DEVICE)

            prediction, z_target = model(flux, mask)

            loss = jepa_loss(prediction, z_target)

            val_total_loss += loss.item()

            if batch_idx % 10 == 0: #why don't we just change this to be like every 30 or so
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, val loss: {loss.item():.6f}")

    average_loss_val = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, average val loss: {average_loss_val:.6f}")


    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/training_physics_jepa.pth')
