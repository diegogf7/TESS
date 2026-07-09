import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")


BATCH_SIZE = 256


#run_probe from claude code
def run_probe(X, y, name, groups=None):
    uni, counts = np.unique(y, return_counts=True)
    keep = uni[counts >= 2]
    m = np.isin(y, keep)
    X, y = X[m], y[m]
    if groups is not None:
        groups = groups[m]

    if groups is not None:
        
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    probe = KNeighborsClassifier(n_neighbors=20)
    probe.fit(x_train, y_train)
    acc = balanced_accuracy_score(y_test, probe.predict(x_test))
    chance = 1 / len(np.unique(y))
    print(f"{name}: balanced accuracy = {acc:.6f} (chance = {chance:.4f})")


print(f"device: {DEVICE}")
dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)
print(f"{len(dataset)} samples, {len(loader)} batches")

model = build_instrument_jepa().to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()


latents = []
all_sectors = []
all_labels = []

with torch.no_grad():
    for i, (flux, mask, sector, label) in enumerate(loader):

        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        z = model.encode(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        all_labels.append(label.numpy())

        if i % 10 == 0:
            print(f"getting batch {i} / {len(loader)}")
    
    latents = np.concatenate(latents)
    all_sectors = np.concatenate(all_sectors)
    all_labels = np.concatenate(all_labels)

    all_tics = dataset.df["TIC"].to_numpy()


print(f"Mean std of latent (collapse check, want >> 0): {latents.std(0).mean():.6f}")

run_probe(latents, all_sectors, "z_inst --> SECTOR (want HIGH, bar 0.657)")
run_probe(latents, all_labels,  "z_inst --> CLASS  (want LOW, generic encoder ~0.41)", groups=all_tics)