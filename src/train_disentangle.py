import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from data import LightCurveDataset, ClassificationDataset, DisentanglementDataset
from s4d import S4Model
from disentangle import DisentanglementModel, reconstruction_loss, consistency_loss, cross_reconstruction_loss

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"


dataset = DisentanglementDataset(DATA_PATH, grid_length = 1024)
val_dataset = DisentanglementDataset(VAL_PATH, grid_length = 1024)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

model = DisentanglementModel(256, 4, 0.2).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)



for epoch in range(EPOCHS):
    model.train()
    total_loss = 0



    for batch_idx, (anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask) in enumerate(dataloader):

        anchor_flux = anchor_flux.to(DEVICE)
        anchor_mask = anchor_mask.to(DEVICE)
        same_star_flux = same_star_flux.to(DEVICE)
        same_star_mask = same_star_mask.to(DEVICE)
        same_sector_flux = same_sector_flux.to(DEVICE)
        same_sector_mask = same_sector_mask.to(DEVICE)

        flux = flux.unsqueeze(-1)

        optimizer.zero_grad()

        #forwards process now
        

        reconstruction, x_physics, x_instruments, x_physics_same_star, x_instrument_same_sector = model(anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask)

        loss = reconstruction_loss(reconstruction, anchor_flux) + consistency_loss(x_physics, x_physics_same_star) + cross_reconstruction_loss(x_physics, x_instrument_same_sector, model.decoder, anchor_flux)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()



        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
        
    average_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {average_loss:.6f}, Balanced accuracy: {balanced_accuracy}")

    scheduler.step()

    model.eval()

    val_total_loss = 0

    with torch.no_grad():

        for batch_idx, (flux, mask, labels) in enumerate(val_dataloader):
            anchor_flux = anchor_flux.to(DEVICE)
            anchor_mask = anchor_mask.to(DEVICE)
            same_star_flux = same_star_flux.to(DEVICE)
            same_star_mask = same_star_mask.to(DEVICE)
            same_sector_flux = same_sector_flux.to(DEVICE)
            same_sector_mask = same_sector_mask.to(DEVICE)

            flux = flux.unsqueeze(-1)

            optimizer.zero_grad()

            #forwards process now
            

            reconstruction, x_physics, x_instruments, x_physics_same_star, x_instrument_same_sector = model(anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask)

            loss = reconstruction_loss(reconstruction, anchor_flux) + consistency_loss(x_physics, x_physics_same_star) + cross_reconstruction_loss(x_physics, x_instrument_same_sector, model.decoder, anchor_flux)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
            
    average_loss_val = val_total_loss / len(val_dataloader)



    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/disentangle.pth')



