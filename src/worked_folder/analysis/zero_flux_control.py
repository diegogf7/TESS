import os 

import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")



BATCH_SIZE = 256
GRID = 1024

dataset = DualEvalDataset(DATA_PATH, grid_length=1024)


loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

latents_full = []
latents_zero = []
all_sectors = []

with torch.no_grad():

    for i, (flux, mask, sector, label) in enumerate(loader):

        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        z_full = model.encode(flux, mask)
        z_zero = model.encode(torch.zeros_like(flux), mask)

        latents_full.append(z_full.reshape(z_full.shape[0], -1).cpu().numpy())
        latents_zero.append(z_zero.reshape(z_zero.shape[0], -1).cpu().numpy())

        all_sectors.append(sector.numpy())


X_full = np.concatenate(latents_full)
X_zero = np.concatenate(latents_zero)
sectors = np.concatenate(all_sectors)

unique, counts = np.unique(sectors, return_counts = True)
keep = np.isin(sectors, unique[counts >= 5])


X_full = X_full[keep]
X_zero = X_zero[keep]
sectors = sectors[keep]

index = np.arange(len(sectors))

index_train, index_test = train_test_split(index, test_size = 0.2, stratify = sectors, random_state = 0)

def probe_accuracy(X):
    scaler = StandardScaler().fit(X[index_train])
    probe = LogisticRegression(max_iter = 2000, class_weight = "balanced").fit(scaler.transform(X[index_train]), sectors[index_train])

    return balanced_accuracy_score(sectors[index_test], probe.predict(scaler.transform(X[index_test])))

accuracy_full = probe_accuracy(X_full)
accuracy_zero = probe_accuracy(X_zero)

print(f"full latent -> SECTOR: {accuracy_full:.5f}")
print(f"zero flux latent -> SECTOR: {accuracy_zero:.5f}")
print(f"flux adds: {accuracy_full - accuracy_zero:.5f}")

