"""
Evaluate the frozen MaskedS4D latent on the real TESS curves (cluster).

Same probe protocol as eval_physics_dann.py: freeze the encoder, extract the
latent, KNN-probe with balanced accuracy. SECTOR is the headline metric here.

    python -m src.bot_folder.eval_masked
"""

import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.masked_s4d import build_masked_s4d

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("MASKED_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/masked_s4d.pth")
BATCH_SIZE = 256


def run_probe(X, y, name):
    uni, counts = np.unique(y, return_counts=True)
    keep = uni[counts >= 2]
    m = np.isin(y, keep)
    X, y = X[m], y[m]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
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
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=4)
print(f"{len(dataset)} samples, {len(loader)} batches")

model = build_masked_s4d().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

latents, all_sectors, all_labels = [], [], []
with torch.no_grad():
    for i, (flux, mask, sector, label) in enumerate(loader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        z = model.encode(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        all_labels.append(label.numpy())
        if i % 10 == 0:
            print(f"  extracting batch {i}/{len(loader)}")

latents = np.concatenate(latents)
all_sectors = np.concatenate(all_sectors)
all_labels = np.concatenate(all_labels)

print(f"Mean std of latent (collapse check, want >> 0): {latents.std(0).mean():.6f}")

run_probe(latents, all_sectors, "latent --> SECTOR (target, want high)")
run_probe(latents, all_labels,  "latent --> CLASS")
