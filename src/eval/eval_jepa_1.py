import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data.data import normalize, resample_to_grid, LightCurveDataset
from src.models.model import masking_autoencoder
from src.data.data import DisentanglementDataset, ClassificationDataset, SectorDataset
from torch.utils.data import DataLoader
from src.models.disentangle import DisentanglementModel, flow_matching_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.jepa_1 import JEPA_1



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "/work1/jeroenaudenaert/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/work1/jeroenaudenaert/diegogon/checkpoints/training_jepa1.pth"
BATCH_SIZE = 256

model = JEPA_1(256, 4, 0.2, 0.996).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

dataset = SectorDataset(DATA_PATH, 1024)
eval_loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 4)

online_latents = []
target_latents = []
all_sectors = []

with torch.no_grad():

    for flux, mask, sector in eval_loader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z_online = model.online_encoder(flux, mask)
        z_target = model.target_encoder(flux, mask)

        online_latents.append(z_online.reshape(z_online.shape[0], -1).cpu().numpy())
        target_latents.append(z_target.reshape(z_target.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
    
online_latents = np.concatenate(online_latents)
target_latents = np.concatenate(target_latents)
all_sectors = np.concatenate(all_sectors)

X_train, X_test, y_train, y_test = train_test_split(target_latents, all_sectors, test_size =0.2, stratify = all_sectors, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

probe = LogisticRegression(max_iter = 5000)
probe.fit(X_train, y_train)

predictions = probe.predict(X_test)

n_sectors = len(np.unique(all_sectors))
print(f"target encoder had a balanced accuracy: {balanced_accuracy_score(y_test, predictions)}")
print(f"the chance for guessing is {1/n_sectors:.6f}")

