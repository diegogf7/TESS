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

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GroupKFold

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe20k_area.parquet")

CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_gapblind_chip_s0.pth")

BATCH_SIZE = 256
MIN_PER_CLASS = 10

N_INFILLS = 5
N_REPEATS = 10
PCA_DIM = 32

dataset = SectorDataset(DATA_PATH, grid_length = 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

sectors = dataset.df["sector"].to_numpy()
camera = dataset.df["camera"].to_numpy()
ccd = dataset.df["ccd"].to_numpy()

cam_ccd = camera * 10 + ccd

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

pieces = []
with torch.no_grad():

    for i, (flux, mask, sector) in enumerate(loader):
        flux = flux.to(DEVICE)

        mask = mask.to(DEVICE)
        z_sum = 0.0

        for _ in range(N_INFILLS):

            filled, _ = infill_gaps(flux, mask)
            z_sum = z_sum + model.encode(filled, None)
        
        z = z_sum / N_INFILLS

        pieces.append(z.reshape(z.shape[0], -1).cpu().numpy())

        if i % 20 == 0:
            print(f"Batch {i} / {len(loader)}")


X = np.concatenate(pieces)

def fit_probe(X_train, y_train, X_test, y_test):

    probe = make_pipeline(
        StandardScaler(),
        PCA(n_components = PCA_DIM, random_state = 0),
        LogisticRegression(max_iter = 2000, class_weight = "balanced"),

    )
    probe.fit(X_train, y_train)
    final = balanced_accuracy_score(y_test, probe.predict(X_test))
    return final


def probe_global(y, label):
    
    scores = []
    for r in range(N_REPEATS):
        

def within_sector_probe(y, label):

    scores = []
    skipped = 0

    for sector in np.unique(sectors):
        in_sector = sectors == sector

        y_sector = y[in_sector]

        classes, counts = np.unique(y_sector, return_counts = True)

        if ((len(classes) < 2) or (counts.min() < MIN_PER_CLASS)):
            skipped = skipped + 1
            continue

        X_sector = X[in_sector]

        index = np.arange(len(y_sector))
        index_train, index_test = train_test_split(index, test_size = 0.2, stratify = y_sector, random_state = 0)
        accuracy = fit_probe(X_sector[index_train], y_sector[index_train], X_sector[index_test], y_sector[index_test])

        scores.append(accuracy)

        print(f"sector {sector}: n = {in_sector.sum()}, {len(classes)} classes, balanced accuracy {accuracy:.5f}")
    
    if not scores:
        print(f"Sector {label}: no sector had enough samples: all {skipped} skipped")
        return
    
    print(f"Sector {label}: mean {np.mean(scores):.5f} over {len(scores)} sectors, {skipped} skipped ")

print(f"checkpoint: {CHECKPOINT}")
print(f"data: {DATA_PATH} ({len(dataset)} curves, {len(np.unique(sectors))} sectors)")

for label, y in [("CAMERA", camera), ("CCD", ccd), ("CAM-CCD", cam_ccd)]:
    print(f"{label}: chance ~ {1.0 / len(np.unique(y)):.5f}")
    within_sector_probe(y, label)
