"""
UMAP of the trained MaskedS4D latent (cluster), colored by SECTOR and by CLASS.
Visual companion to eval_masked.py — shows the structure behind the probe numbers.

    python -m src.bot_folder.umap_masked
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.data import DualEvalDataset, CLASSES
from src.bot_folder.masked_s4d import build_masked_s4d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/masked_s4d.pth"
OUT_DIR = "/orcd/scratch/orcd/006/diegogon/checkpoints"
BATCH_SIZE = 256

model = build_masked_s4d().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)

latents, sectors, labels = [], [], []
with torch.no_grad():
    for flux, mask, sector, label in loader:
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        z = model.encode(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        sectors.append(sector.numpy())
        labels.append(label.numpy())

latents = np.concatenate(latents)
sectors = np.concatenate(sectors)
labels = np.concatenate(labels)

X = StandardScaler().fit_transform(latents)
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)

# colored by SECTOR (the target structure)
plt.figure(figsize=(9, 7))
sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=sectors, s=4, alpha=0.3, cmap="tab20")
plt.colorbar(sc, label="sector")
plt.title("Masked-S4D latent UMAP (colored by SECTOR)")
plt.savefig(f"{OUT_DIR}/masked_umap_by_sector.png", dpi=150, bbox_inches="tight")
plt.close()

# colored by CLASS
plt.figure(figsize=(9, 7))
for c in np.unique(labels):
    m = labels == c
    plt.scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.3, label=CLASSES[c])
plt.legend(markerscale=3, fontsize=8)
plt.title("Masked-S4D latent UMAP (colored by CLASS)")
plt.savefig(f"{OUT_DIR}/masked_umap_by_class.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"saved masked_umap_by_sector.png and masked_umap_by_class.png to {OUT_DIR}")
