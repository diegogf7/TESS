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
from src.models.jepa_2 import JEPA_2



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/training_jepa2.pth"
BATCH_SIZE = 256

model = JEPA_2(256, 4, 0.2, 0.996).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

dataset = ClassificationDataset(DATA_PATH, 1024)
eval_loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 4)


target_latents = []
all_labels = []

with torch.no_grad():

    for flux, mask, label in eval_loader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z = model.target_encoder(flux, mask)
        target_latents.append(z.reshape(z.shape[0], -1).cpu().numpy())

        all_labels.append(label.numpy())

    
target_latents = np.concatenate(target_latents)
all_labels = np.concatenate(all_labels)

keep = all_labels != 5
target_latents = target_latents[keep]
all_labels = all_labels


X_train, X_test, y_train, y_test = train_test_split(target_latents, all_labels, test_size =0.2, stratify = all_labels, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

probe = LogisticRegression(max_iter = 5000)
probe.fit(X_train, y_train)

predictions = probe.predict(X_test)

n_classes = len(np.unique(all_labels))
print(f"target encoder had a balanced accuracy: {balanced_accuracy_score(y_test, predictions)}")
print(f"the chance for guessing is {1/n_classes:.6f}")

