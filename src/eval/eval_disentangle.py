import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data.data import normalize, resample_to_grid, LightCurveDataset
from src.models.model import masking_autoencoder
from src.data.data import DisentanglementDataset, ClassificationDataset
from torch.utils.data import DataLoader
from src.models.disentangle import DisentanglementModel, flow_matching_loss
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



BATCH_SIZE = 256
EPOCHS = 100
TEST_PATH  = "/work1/jeroenaudenaert/diegogon/phyts/TESS/TESS/split/tess_classification_test.parquet"
DATA_PATH = "/work1/jeroenaudenaert/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DisentanglementModel().to(DEVICE)
model.load_state_dict(torch.load('/work1/jeroenaudenaert/diegogon/checkpoints/disentangle.pth', map_location=DEVICE))
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
        z_physics = z_physics.reshape(z_physics.shape[0], -1) #reducing one dimension

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
        z_physics = z_physics.reshape(z_physics.shape[0], -1) #reducing one dimension


        val_latents.append(z_physics.cpu().numpy())
        val_labels.append(labels.numpy())

val_latents = np.concatenate(val_latents)
val_labels = np.concatenate(val_labels)


scaler = StandardScaler()
train_latents = scaler.fit_transform(train_latents)
val_latents = scaler.transform(val_latents)
viewing_classifier = LogisticRegression(max_iter = 5000)
viewing_classifier.fit(train_latents, train_labels)
predictions = viewing_classifier.predict(val_latents)
print(f"Balanced accuracy score: {balanced_accuracy_score(val_labels, predictions)}")
