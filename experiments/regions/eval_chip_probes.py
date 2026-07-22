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
        train, test = train_test_split(np.arange(len(y)), test_size = 0.2, stratify = y, random_state = r)
        scores.append(fit_probe(X[train], y[train], X[test], y[test]))

    print(f"Global {label}: {np.mean(scores):.5f} plus/minus {np.std(scores):.4f} ({N_REPEATS} splits)")


def within_sector_probe(y, label):

    sector_means = []
    skipped_values = 0

    for sector in np.unique(sectors):
        in_sector = sectors == sector
        y_sector = y[in_sector]
        X_sector = X[in_sector]

        classes, counts = np.unique(y_sector, return_counts = True)
        if (len(classes) < 2) or (counts.min() < MIN_PER_CLASS):

            skipped_values = skipped_values + 1
            continue

        scores = []
        for r in range(5):

            train, test = train_test_split(np.arange(len(y_sector)), test_size = 0.2, stratify = y_sector, random_state = r)
            scores.append(fit_probe(X_sector[train], y_sector[train], X_sector[test], y_sector[test]))

        sector_means.append(np.mean(scores))

    if not sector_means:
        print(f"Within sector {label}: all {skipped_values} sectors skipped")
        return
    
    print(f"Within sector {label}: {np.mean(sector_means):.5f} plus/minus {np.std(sector_means):.5f} over {len(sector_means)} sectors, {skipped_values} skipped")

def leave_sectors_out_probe(y, label):

    scores = []
    for train, test in GroupKFold(n_splits = 5).split(X, y, groups = sectors):
        scores.append(fit_probe(X[train], y[train], X[test], y[test]))

    print(f"Leave sector out {label}: {np.mean(scores):.5f} plus/minus {np.std(scores):.5f} (5 folds)")


print(f"Checkpoint: {CHECKPOINT}")
print(f"Data: {DATA_PATH} ({len(dataset)} curves, {len(np.unique(sectors))} sectors)")

for label, y in [("CAMERA", camera), ("CCD", ccd), ("CAM-CCD", cam_ccd)]:

    print(f" {label}: chance ~ {1.0 / len(np.unique(y)):.5f}")
    probe_global(y, label)
    within_sector_probe(y, label)
    
    leave_sectors_out_probe(y, label)
