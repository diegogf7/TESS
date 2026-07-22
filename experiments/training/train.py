import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.data import LightCurveDataset
from src.models.model import masking_autoencoder


#global variables

BATCH_SIZE = 256
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/tess_classification.parquet"

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"

print("Loading dataset...", flush=True)
dataset = LightCurveDataset(DATA_PATH, grid_length = 1024)
val_dataset = LightCurveDataset(VAL_PATH, grid_length = 1024)
print(f"Loaded {len(dataset)} light curves", flush=True)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

model = masking_autoencoder().to(DEVICE)
print(f"Model ready on {DEVICE}", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)

#need to make a schedule so that we get the most changes initially
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

criteria = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (flux, mask) in enumerate(dataloader):
        flux = flux.to(DEVICE)

        optimizer.zero_grad()

        #forward
        predictions, shuffle = model(flux)

        #the backward
        #need to change the size of the flux
        targets = flux.reshape(-1, 64, 16)

        masked_indexes = shuffle[:, 16:].unsqueeze(-1).expand(-1, -1, 16)
        prediction_masked = torch.gather(predictions, 1, masked_indexes)
        true_masked = torch.gather(targets, 1, masked_indexes)
        loss = criteria(prediction_masked, true_masked)

        loss.backward()
        #and the backprop
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch+1}, batch {batch_idx}/{len(dataloader)}, loss: {loss.item():.4f}", flush=True)



    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{EPOCHS} complete — Avg Loss: {average_loss:.4f}', flush=True)
    scheduler.step()

    model.eval()

    val_total_loss = 0
    with torch.no_grad():
        for batch_idx, (flux, mask) in enumerate(val_dataloader):
            flux = flux.to(DEVICE)
            
            predictions, shuffle = model(flux)

            targets = flux.reshape(-1, 64, 16)

            masked_indexes = shuffle[:, 16:].unsqueeze(-1).expand(-1, -1, 16)
            prediction_masked = torch.gather(predictions, 1, masked_indexes)
            true_masked = torch.gather(targets, 1, masked_indexes)
            loss = criteria(prediction_masked, true_masked)

            val_total_loss += loss.item()
            if batch_idx % 10 == 0:

                print(f"  Epoch {epoch+1}, batch {batch_idx}/{len(val_dataloader)}, loss: {loss.item():.4f}", flush=True)
    
    average_val_loss = val_total_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{EPOCHS} complete — Avg Loss: {average_val_loss:.4f}', flush=True)




    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/mae_final.pth')

