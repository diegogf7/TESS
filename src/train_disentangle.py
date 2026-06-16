import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
import numpy as np


from data import LightCurveDataset, ClassificationDataset, DisentanglementDataset
from s4d import S4Model
from disentangle import DisentanglementModel, flow_matching_loss, info_nce

def random_mask(flux, mask, mask_ratio = 0.3, random_ratio = 0.1, true_ratio = 0.1):


    #we have to do different baskets for the masking
    
    first_bucket = torch.rand(mask.shape, device = mask.device)

    masked = first_bucket < mask_ratio
    random_part = (first_bucket >= mask_ratio) & (first_bucket < mask_ratio + random_ratio) #excluding it at the end

    new_flux = flux.clone()
    new_flux[masked] = 0.0
    new_flux[random_part] = torch.rand_like(new_flux[random_part])

    new_mask = mask.clone()
    new_mask[masked] = 0.0

    return new_flux, new_mask




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

optimizer = torch.optim.AdamW(model.parameters(), lr =0.001)
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

        
        same_star_flux = same_star_flux.unsqueeze(-1)
        same_sector_flux = same_sector_flux.unsqueeze(-1)

        same_star_flux, same_star_mask = random_mask(same_star_flux, same_star_mask, mask_ratio = 0.3, random_ratio = 0.1, true_ratio = 0.1)
        same_sector_flux, same_sector_mask = random_mask(same_sector_flux, same_sector_mask, mask_ratio = 0.3, random_ratio = 0.1, true_ratio = 0.1)

        optimizer.zero_grad()

        #forwards process now
        

        x_physics, x_instrument = model(same_star_flux, same_star_mask, same_sector_flux, same_sector_mask)
        concatenate = torch.cat([x_physics, x_instrument], dim = 1)

        z_phys_view = x_physics.flatten(1)
        z_phys_anchor = model.physics_encoder(anchor_flux.unsqueeze(-1), anchor_mask).flatten(1)
        contrastive = info_nce(z_phys_anchor, z_phys_view)

        loss = (0.2 * contrastive) + flow_matching_loss(model.velocity_net, anchor_flux, concatenate)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()



        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
        
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} / {EPOCHS}, average loss: {average_loss:.6f}")

    scheduler.step()

    model.eval()

    val_total_loss = 0

    with torch.no_grad():

        for batch_idx, (anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask) in enumerate(val_dataloader):
            anchor_flux = anchor_flux.to(DEVICE)
            anchor_mask = anchor_mask.to(DEVICE)
            same_star_flux = same_star_flux.to(DEVICE)
            same_star_mask = same_star_mask.to(DEVICE)
            same_sector_flux = same_sector_flux.to(DEVICE)
            same_sector_mask = same_sector_mask.to(DEVICE)

            
            same_star_flux = same_star_flux.unsqueeze(-1)
            same_sector_flux = same_sector_flux.unsqueeze(-1)


            #forwards process now
            

            x_physics, x_instrument = model(same_star_flux, same_star_mask, same_sector_flux, same_sector_mask)

            concatenate = torch.cat([x_physics, x_instrument], dim=1)

            loss = flow_matching_loss(model.velocity_net, anchor_flux, concatenate)

            val_total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
            
    average_loss_val = val_total_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} / {EPOCHS}, average loss: {average_loss_val:.6f}")



    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/disentangle.pth')

