import torch

import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import umap

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.data import DualEvalDataset, CLASSES
from src.models.physics_jepa import PhysicsJEPA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/training_physics_jepa.pth"
OUT_DIR    = "/orcd/scratch/orcd/006/diegogon/checkpoints"
BATCH_SIZE = 256

model = PhysicsJEPA(256, 4, 0.2, 0.996).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))

model.eval()

dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 4)

#collecting for matplotlib
latents = []
sectors = []
labels = []

with torch.no_grad():
    for flux, mask, sector, label in loader:

        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)
        z = model.target_encoder(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        sectors.append(sector.numpy())
        labels.append(label.numpy())

latents = np.concatenate(latents)
sectors = np.concatenate(sectors)
labels = np.concatenate(labels)

#now I need the UMAP to be in 2D 

X = StandardScaler().fit_transform(latents)
embedding = umap.UMAP(n_neighbors = 15, min_dist = 0.1, random_state = 0).fit_transform(X)

#now I need it by class
plt.figure(figsize = (9, 7))
for c in np.unique(labels):
    m = labels == c #what is this doing
    plt.scatter(embedding[m, 0], embedding[m, 1], s=4, alpha = 0.2, label = CLASSES[c])

plt.legend(markerscale =3, fontsize = 8)
plt.title("Latent UMAP (Colored by our Classes)")
plt.savefig(f"{OUT_DIR}/umap_by_class.png", dpi=150, bbox_inches="tight")
plt.close()

#now I need it by sector
plt.figure(figsize = (9,7))
sc = plt.scatter(embedding[:, 0], embedding[:, 1], c = sectors, s=4, alpha = 0.2, cmap = "viridis")
plt.colorbar(sc, label = "sector")

plt.title("Latent UMAP (Colored by our Sectors)")
plt.savefig(f"{OUT_DIR}/umap_by_sector.png", dpi=150, bbox_inches="tight")
plt.close()

df = pd.read_parquet(DATA_PATH)
lengths = df["flux"].apply(len).to_numpy()

plt.figure(figsize = (9, 7))
sc = plt.scatter(embedding[:,0], embedding[:, 1], c = lengths, s=3, alpha = 0.2, cmap = "viridis")
plt.colorbar(sc, label = "light curve length")
plt.title("Latent UMAP (Colored by light curve length)")
plt.savefig(f"{OUT_DIR}/umap_by_length.png", dpi=150, bbox_inches="tight")
plt.close()


print(f"saved umap_by_class.png, umap_by_sector.png, umap_by_length.png to {OUT_DIR}")
