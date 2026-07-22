import torch
import torch.nn as nn

from src.data.data import DisentanglementDataset, DataLoader
from src.models.jepa_2 import JEPA_2, jepa_loss

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"


dataset = DisentanglementDataset(DATA_PATH, grid_length =1024, multi_sector_only=True)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

val_dataset = DisentanglementDataset(VAL_PATH, grid_length = 1024, multi_sector_only = True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)


model = JEPA_2(d_model = 256, n_layers =4, dropout = 0.2, momentum = 0.996).to(DEVICE)
#we're only going to train our online encoder not the EMA

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

for epoch in range(EPOCHS):

    model.train()
    total_loss= 0

    for batch_idx, (anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask) in enumerate(dataloader):
        anchor_flux = anchor_flux.to(DEVICE)
        anchor_mask = anchor_mask.to(DEVICE)
        same_star_flux = same_star_flux.to(DEVICE)
        same_star_mask = same_star_mask.to(DEVICE)

        anchor_flux = anchor_flux.unsqueeze(-1)
        same_star_flux = same_star_flux.unsqueeze(-1)

        optimizer.zero_grad()

        #context is the different star and then the target is the anchor (so flipped)

        prediction, z_target = model(same_star_flux, same_star_mask, anchor_flux, anchor_mask)

        loss = jepa_loss(prediction, z_target)

        loss.backward()
        optimizer.step()
        model.update_target() #for the EMA

        total_loss = total_loss + loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
    
    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {total_loss/len(dataloader):.6f}")
    scheduler.step()



    #moving to the validation step now
    model.eval()

    val_total_loss = 0

    with torch.no_grad():

        for batch_idx, (anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask) in enumerate(val_dataloader):
            anchor_flux = anchor_flux.to(DEVICE)
            anchor_mask = anchor_mask.to(DEVICE)
            same_star_flux = same_star_flux.to(DEVICE)
            same_star_mask = same_star_mask.to(DEVICE)

            anchor_flux = anchor_flux.unsqueeze(-1)
            same_star_flux = same_star_flux.unsqueeze(-1)

            prediction, z_target = model(same_star_flux, same_star_mask, anchor_flux, anchor_mask)


            #forwards process now
            loss = jepa_loss(prediction, z_target)

            val_total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
            
    average_loss_val = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} / {EPOCHS}, average loss: {average_loss_val:.6f}")
    

    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/training_jepa2.pth')