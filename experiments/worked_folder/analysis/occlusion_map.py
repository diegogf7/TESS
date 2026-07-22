import os

import matplotlib
matplotlib.use("Agg")  # headless: cluster nodes have no display
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")
OUT_DIR = os.path.dirname(CHECKPOINT)
BATCH_SIZE = 256

GRID = 1024
WIDTH = 64
STRIDE = 32
OCCLUSION_CURVES = 200
N_SHOW = 6
N_RANDOM = 5

positions = list(range(0, GRID - WIDTH + 1, STRIDE))

n_win = len(positions)
random_number = np.random.default_rng(0)

dataset = DualEvalDataset(DATA_PATH, GRID)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()


latents = []
all_sectors = []
all_flux = []
all_mask = []

with torch.no_grad():

    for i, (flux, mask, sector, label) in enumerate(loader):

        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        z = model.encode(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        all_flux.append(flux.cpu().numpy())
        all_mask.append(mask.cpu().numpy())

        if i % 10 == 0:
            print(f" extracting batch {i} / {len(loader)}")

all_latents = np.concatenate(latents)
sectors = np.concatenate(all_sectors)
fluxes = np.concatenate(all_flux)
masks = np.concatenate(all_mask)

unique, counts = np.unique(sectors, return_counts = True)
keep = np.isin(sectors, unique[counts >= 2])
all_latents = all_latents[keep]
sectors = sectors[keep]
fluxes = fluxes[keep]
masks = masks[keep]

index = np.arange(len(all_latents))
index_train, index_test = train_test_split(index, test_size = 0.2, stratify = sectors, random_state = 0)

scaler = StandardScaler().fit(all_latents[index_train])
probe = LogisticRegression(max_iter = 2000).fit(scaler.transform(all_latents[index_train]), sectors[index_train])

accuracy = balanced_accuracy_score(sectors[index_test], probe.predict(scaler.transform(all_latents[index_test])))

print(f"probe balanced accuracy = {accuracy:.4f} (sanity check)")

occlusion_index = random_number.choice(index_test, size = min(OCCLUSION_CURVES, len(index_test)), replace = False)

importance = np.zeros((len(occlusion_index), n_win))
base_p = np.zeros(len(occlusion_index))

random_drops = []

with torch.no_grad():

    for k, di in enumerate(occlusion_index):

        flux = torch.from_numpy(fluxes[di])
        mask = torch.from_numpy(masks[di])

        true_column = int(np.where(probe.classes_ == sectors[di])[0][0])

        variants = flux.unsqueeze(0).repeat(1 + n_win + N_RANDOM, 1).clone()
        for w, pos in enumerate(positions):

            variants[1 + w, pos: pos + WIDTH] = 0.0
        
        for r in range(N_RANDOM):

            scatter = random_number.choice(GRID, size = WIDTH, replace = False)
            variants[1 + n_win + r, scatter] = 0.0

        mask_batch = mask.unsqueeze(0).repeat(1 + n_win + N_RANDOM, 1)
        z = model.encode(variants.to(DEVICE), mask_batch.to(DEVICE))

        z = z.reshape(z.shape[0], -1).cpu().numpy()

        p = probe.predict_proba(scaler.transform(z))[:, true_column]
        
        base_p[k] = p[0]
        importance[k] = p[0] - p[1: 1 + n_win]
        random_drops.append(p[0] - p[1 + n_win:])

        if k % 30 == 0:
            print(f"  occluding curve {k} / {len(occlusion_index)}")

random_drops = np.concatenate(random_drops)
print(f"mean baseline P(true sector) = {base_p.mean():.5f}")

#now we add the other importance and scatter information
print(f"mean window importance = {np.abs(importance).mean():.5f}")
print(f"mean scatter importance = {np.abs(random_drops).mean():.5f}")

for k in range(min(N_SHOW, len(occlusion_index))):

    di = occlusion_index[k]
    #now need to see the importance along the whole grid

    point_importance = np.zeros(GRID)
    cover = np.zeros(GRID)

    for w, position in enumerate(positions):
        point_importance[position:position + WIDTH] += importance[k, w]
        
        cover[position: position + WIDTH] += 1
    
    point_importance = point_importance / np.maximum(cover, 1)
    
    figure, (ax0, ax1) = plt.subplots(2, 1, figsize = (11, 5), sharex = True, gridspec_kw = {"height_ratios": [2,1]})

    ax0.plot(fluxes[di], ".", ms=3)
    ax0.set_ylabel("flux normalized")
    ax0.set_title(f"sector {sectors[di]} baseline P = {base_p[k]:.3f}", fontsize = 9)
    ax1.fill_between(np.arange(GRID), point_importance, 0.0, color = "blue", alpha = 0.4)
    ax1.axhline(0.0, color = "gray", lw = 0.5)

    ax1.set_ylabel("importance (drop in P)")
    ax1.set_xlabel("grid index")

    figure.tight_layout()
    figure.savefig(os.path.join(OUT_DIR, f"occlusion_curve_{k}.png"), dpi = 150)
    plt.close(figure)

occlusion_sectors = sectors[occlusion_index]
sector_unique = np.unique(occlusion_sectors)

mean_maps = np.stack([importance[occlusion_sectors == s].mean(axis= 0) for s in sector_unique])

figure, axis =plt.subplots(figsize = (10, 0.35 * len(sector_unique) + 2))

vmax = np.abs(mean_maps).max()

importa = axis.imshow(mean_maps, aspect = "auto", cmap = "RdBu_r", vmin = -vmax, vmax = vmax, extent = (0, GRID, len(sector_unique) - 0.5, -0.5))

axis.set_yticks(range(len(sector_unique)))
axis.set_yticklabels(sector_unique, fontsize = 5)
axis.set_xlabel("grid index")
axis.set_ylabel("sector")

axis.set_title("mean occlusion per sector")
figure.colorbar(importa, ax = axis, fraction = 0.025)
figure.savefig(os.path.join(OUT_DIR, "occlusion_sector_mean.png"), dpi = 150)
