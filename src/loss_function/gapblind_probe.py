"""Decompose the instrument latent's sector signal: mask-borne vs flux-borne.

Encodes the same eval curves three ways and probes SECTOR from each:

  full      : real flux + real mask          (what eval normally sees)
  zero-flux : zeroed flux + real mask        (mask-borne signal ONLY)
  gap-blind : infilled flux + no mask        (flux-borne signal ONLY)

Reading the result: gap-blind is the number that matters. If it is well above
chance, the encoder carries real flux systematics. If it sits at chance, all
sector information is coming from gap placement.

    JEPA_CKPT=/path/to/checkpoint.pth python -m src.loss_function.gapblind_probe
"""

import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.loss_function.gap_infill import infill_gaps
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet")
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_gapmask.pth")

BATCH_SIZE = 256

print(f"device: {DEVICE}")
print(f"checkpoint: {CHECKPOINT}")

dataset = DualEvalDataset(DATA_PATH, grid_length=1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

latents = {"full": [], "zero_flux": [], "gap_blind": []}
all_sectors = []

with torch.no_grad():
    for i, (flux, mask, sector, label) in enumerate(loader):
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

# same filtering + split as zero_flux_control so numbers stay comparable
unique, counts = np.unique(sectors, return_counts=True)
keep = np.isin(sectors, unique[counts >= 5])
sectors = sectors[keep]
latents = {name: X[keep] for name, X in latents.items()}

index = np.arange(len(sectors))
index_train, index_test = train_test_split(index, test_size=0.2, stratify=sectors, random_state=0)


def probe_accuracy(X):
    scaler = StandardScaler().fit(X[index_train])
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X[index_train]), sectors[index_train])
    return balanced_accuracy_score(sectors[index_test], probe.predict(scaler.transform(X[index_test])))


chance = 1.0 / len(np.unique(sectors))
accuracy = {name: probe_accuracy(X) for name, X in latents.items()}

print(f"\ncheckpoint: {CHECKPOINT}")
print(f"{len(np.unique(sectors))} sectors, chance = {chance:.4f}")
print(f"full      (flux + mask) -> SECTOR: {accuracy['full']:.5f}")
print(f"zero-flux (mask only)   -> SECTOR: {accuracy['zero_flux']:.5f}   <- mask-borne")
print(f"gap-blind (flux only)   -> SECTOR: {accuracy['gap_blind']:.5f}   <- flux-borne, the number that matters")
