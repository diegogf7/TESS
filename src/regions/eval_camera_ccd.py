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

DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet")
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_raw_edges.pth")

BATCH_SIZE = 256
print(f"data: {DATA_PATH}")
print(f"checkpoint: {CHECKPOINT}")

dataset = SectorDataset(DATA_PATH, grid_length = 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)
camera = dataset.df["camera"].to_numpy()

ccd = dataset.df["ccd"].to_numpy()

cam_ccd = camera * 10 + ccd

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))

model.eval()

latents = {"full": [], "zero_flux": [], "gap_blind": []}

with torch.no_grad():

    for i, (flux, mask, sector) in enumerate(loader):
        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)

        z_full = model.encode(flux, mask)
        z_zero = model.encode(torch.zeros_like(flux), mask)

        filled, _ = infill_gaps(flux, mask)
        z_blind = model.encode(filled, None)

        #just need to add some characteristics

        latents["full"].append(z_full.reshape(z_full.shape[0], -1).cpu().numpy())
        latents["zero_flux"].append(z_zero.reshape(z_zero.shape[0], -1).cpu().numpy())
        latents["gap_blind"].append(z_blind.reshape(z_blind.shape[0], -1).cpu().numpy())


        if i % 20 == 0:
            print(f"Batch {i} / {len(loader)}")


latents = {name: np.concatenate(piece) for name, piece in latents.items()}

def probe_accuracy(X, y):

    index = np.arange(len(y))
    index_train, index_test = train_test_split(index, test_size = 0.2, stratify = y, random_state = 0)

    scaler = StandardScaler().fit(X[index_train])
    probe = LogisticRegression(max_iter = 2000, class_weight = "balanced").fit(scaler.transform(X[index_train]), y[index_train])

    return balanced_accuracy_score(y[index_test], probe.predict(scaler.transform(X[index_test])))


print(f"checkpoint: {CHECKPOINT}")

for label, y in [("CAMERA", camera), ("CCD", ccd), ("CAM-CCD", cam_ccd)]:


    chance = 1.0 / len(np.unique(y))

    print(f"{label}: {len(np.unique(y))} classes, chance = {chance:.5f}")

    for variant in ["full", "zero_flux", "gap_blind"]:
        print(f"{variant:10s} -> {probe_accuracy(latents[variant], y):.5f}")