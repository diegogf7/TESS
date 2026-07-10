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
loader = DataLoader(dataset, shuffle = False, num_workers = 2)

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

        