import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import SectorDataset
from src.loss_function.gap_infill import infill_gaps
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val.parquet")
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_raw.pth")

BATCH_SIZE = 256

print(f"data: {DATA_PATH}")
print(f"checkpoint: {CHECKPOINT}")

dataset = SectorDataset(DATA_PATH, grid_length = 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

latents = {"full": [], "zero_flux": [], "gap_blind": []}
all_sectors = []

with torch.no_grad():

    for i, (flux, mask, sector) in enumerate(loader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        z_full = model.encode(flux, mask)
        z_zero = model.encode(torch.zeros_like(flux), mask)


        filled, _ = infill_gaps(flux, mask)
        z_blind = model.encode(filled, None)

        latents["full"].append(z_full.reshape(z_full.shape[0], -1).cpu().numpy())
        latents["zero_flux"].append(z_zero.reshape(z_zero.shape[0], -1).cpu().numpy())

        latents["gap_blind"].append(z_blind.reshape(z_blind.shape[0], -1).cpu().numpy())

        all_sectors.append(sector.numpy())

        if i % 10 == 0:
            print(f"encoding batch {i} / {len(loader)}")

sectors = np.concatenate(all_sectors)
latents = {name: np.concatenate(chunks) for name, chunks in latents.items()}

unique, counts = np.unique(sectors, return_counts = True)

keep = np.isin(sectors, unique[counts >= 5])
sectors = sectors[keep]
latents = {name: X[keep] for name, X in latents.items()}

index = np.arange(len(sectors))
index_train, index_test = train_test_split(index, test_size = 0.2, stratify = sectors, random_state = 0)

def probe_accuracy(X):

    scaler = StandardScaler().fit(X[index_train])
    probe = LogisticRegression(max_iter = 2000, class_weight = "balanced").fit(scaler.transform(X[index_train]), sectors[index_train])

    return balanced_accuracy_score(sectors[index_test], probe.predict(scaler.transform(X[index_test])))

chance = 1.0 /len(np.unique(sectors))

accuracy = {name: probe_accuracy(X) for name, X in latents.items()}

print(f"checkpoint: {CHECKPOINT}")
print(f"{len(np.unique(sectors))} sectors, chance = {chance:.5f}")
print(f"full      (flux + mask) -> SECTOR: {accuracy['full']:.5f}")
print(f"zero-flux (mask only)   -> SECTOR: {accuracy['zero_flux']:.5f}   <- mask-borne")
print(f"gap-blind (flux only)   -> SECTOR: {accuracy['gap_blind']:.5f}   <- flux-borne, the number that matters")



