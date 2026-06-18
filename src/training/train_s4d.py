import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from src.data.data import LightCurveDataset, ClassificationDataset
from src.models.s4d import S4Model

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"


dataset = ClassificationDataset(DATA_PATH, grid_length = 1024)
val_dataset = ClassificationDataset(VAL_PATH, grid_length = 1024)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

model = S4Model(d_input = 1, d_output = 8, d_model = 256, n_layers = 4, dropout = 0.2).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

criteria = nn.CrossEntropyLoss()


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    all_predictions = []
    all_labels = []

    val_all_predictions = []
    val_all_labels = []

    for batch_idx, (flux, mask, labels) in enumerate(dataloader):

        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)

        flux = flux.unsqueeze(-1)

        optimizer.zero_grad()

        #forwards process now
        

        outputs = model(flux, mask = mask)

        loss = criteria(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predictions = outputs.argmax(dim = 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())



        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
        
    average_loss = total_loss / len(dataloader)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {average_loss:.6f}, Balanced accuracy: {balanced_accuracy}")

    scheduler.step()

    model.eval()

    val_total_loss = 0

    with torch.no_grad():

        for batch_idx, (flux, mask, labels) in enumerate(val_dataloader):
            flux = flux.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)


            flux = flux.unsqueeze(-1)


            #forwards process now
            

            outputs = model(flux, mask = mask)

            loss = criteria(outputs, labels)
            val_total_loss += loss.item()

            predictions = outputs.argmax(dim = 1)
            val_all_predictions.extend(predictions.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())



            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss: {loss.item():.6f}")
            
    average_loss_val = val_total_loss / len(val_dataloader)
    balanced_accuracy = balanced_accuracy_score(val_all_labels, val_all_predictions)

    print(f"Epoch {epoch+1}/{EPOCHS}, average loss: {average_loss_val:.6f}, Balanced accuracy: {balanced_accuracy}")



    torch.save(model.state_dict(), '/orcd/scratch/orcd/006/diegogon/checkpoints/s4d_classification.pth')
