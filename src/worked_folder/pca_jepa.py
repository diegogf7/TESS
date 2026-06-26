import os


os.environ.setdefault("JEPA_NTOKENS", "16")
os.environ.setdefault("JEPA_READOUT", "mean_std")
import torch
import torch.nn as nn
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data.data import DualEvalDataset, CLASSES
from src.worked_folder.latent_jepa import build_latent_jepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth"
OUT_DIR = "/orcd/scratch/orcd/006/diegogon/checkpoints"
BATCH_SIZE = 256
MARKERS = ["o", "s", "D", "^", "p", "H", "*", "X"]

model = build_latent_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

latents = []
sectors = []
labels = []

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

    pca = PCA(n_components = 2, random_state = 0)
    embedding = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    plt.figure(figsize= (9,7))
    for c in np.unique(labels):
        m = labels ==c
        plt.scatter(embedding[m, 0], embedding[m,1], s=18, alpha = 0.2, marker = MARKERS[c], label = CLASSES[c])

    plt.legend(markerscale = 2, fontsize= 10)
    plt.xlabel("PC1 Variance")
    plt.ylabel("PC2 Variance")

    plt.title("JEPA PCA for Class")
    plt.savefig(f"{OUT_DIR}/jepa_pca_ms16_by_class.png", dpi=150, bbox_inches="tight")

    plt.close()

    plt.figure(figsize= (9, 7))

    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c= sectors, s=4, alpha = 0.2, cmap = "tab20")
    plt.colorbar(sc, label ="sector")
    plt.xlabel("PC1 Variance")
    plt.ylabel("PC2 Variance")

    plt.title("Labelt PCA for Sector")

    plt.savefig(f"{OUT_DIR}/jepa_pca_ms16_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"PC1+PC2 explained variance: {ev[0]:.1%} + {ev[1]:.1%} = {ev.sum():.1%}")
    print(f"saved jepa_pca_ms16_by_class.png and jepa_pca_ms16_by_sector.png to {OUT_DIR}")

