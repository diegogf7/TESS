import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import normalize, resample_to_grid, LightCurveDataset
from model import masking_autoencoder
from data import DisentanglementDataset, ClassificationDataset
from torch.utils.data import DataLoader
from disentangle import DisentanglementModel, reconstruction_loss
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegression



BATCH_SIZE = 256
EPOCHS = 100
TEST_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_test.parquet"
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DisentanglementModel().to(DEVICE)
model.load_state_dict(torch.load('/orcd/scratch/orcd/006/diegogon/checkpoints/disentangle.pth', map_location=DEVICE))
model.eval()

dataset = ClassificationDataset(DATA_PATH, grid_length = 1024)

val_dataset = ClassificationDataset(TEST_PATH, grid_length = 1024)

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)
train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)


train_latents = []
train_labels = []

with torch.no_grad():
    for flux, mask, labels in train_dataloader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z_physics = model.physics_encoder(flux, mask)

        train_latents.append(z_physics.cpu().numpy())
        train_labels.append(labels.numpy())
    
train_latents = np.concatenate(train_latents)
train_labels = np.concatenate(train_labels)

val_latents = []
val_labels = []

with torch.no_grad():
    for flux, mask, labels in val_dataloader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z_physics = model.physics_encoder(flux, mask)

        val_latents.append(z_physics.cpu().numpy())
        val_labels.append(labels.numpy())

val_latents = np.concatenate(val_latents)
val_labels = np.concatenate(val_labels)


viewing_classifier = LogisticRegression(max_iter = 1000)
viewing_classifier.fit(train_latents, train_labels)
predictions = viewing_classifier.predict(val_latents)
print(f"Balanced accuracy score: {balanced_accuracy_score(val_labels, predictions)}")
