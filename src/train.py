import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import LightCurveDataset
from model import masking_autoencoder


#global variables

BATCH_SIZE = 256
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/tess_classification.parquet"


dataset = LightCurveDataset(DATA_PATH, grid_length = 1024)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

model = masking_autoencoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)

criteria = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    

    for flux, mask in dataloader:
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

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/200 complete — Avg Loss: {average_loss:.4f}')
    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/mae_final.pth')

        

