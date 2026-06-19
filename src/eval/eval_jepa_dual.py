import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from src.data.data import normalize, resample_to_grid, CLASS_TO_IDX, DualEvalDataset
from src.models.jepa_dual import DualJEPA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/training_jepa_dual.pth"
BATCH_SIZE = 256

def run_probe(X, y, name):
    uni, counts = np.unique(y, return_counts = True)

    keep = uni[counts >= 2]
    m = np.isin(y, keep)
    X, y = X[m], y[m]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    probe = LogisticRegression(max_iter = 5000)
    probe.fit(x_train, y_train)
    predictions = probe.predict(x_test)

    accuracy = balanced_accuracy_score(y_test, predictions)

    chance = 1 /len(np.unique(y))

    print(f"{name}: balanced accuracy = {accuracy: .6f} (chance = {chance:.4f})")

model = DualJEPA(256, 4, 0.2, 0.996).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 4)

physics_latent = []
instrument_latent = []
all_sectors = []
all_labels = []

with torch.no_grad():
    for flux, mask, sector, label in loader:

        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z_physics = model.target_physics(flux, mask)
        z_instrument = model.target_instrument(flux, mask)

        physics_latent.append(z_physics.reshape(z_physics.shape[0], -1).cpu().numpy())
        instrument_latent.append(z_instrument.reshape(z_instrument.shape[0], -1).cpu().numpy())

        all_sectors.append(sector.numpy())
        all_labels.append(label.numpy())


physics_latent = np.concatenate(physics_latent)
instrument_latent = np.concatenate(instrument_latent)
all_sectors = np.concatenate(all_sectors)
all_labels = np.concatenate(all_labels)

print(f"Mean std physics latent {physics_latent.std(0).mean()}")
print(f"Mean std instrument latent {instrument_latent.std(0).mean()}")


#need to review this part here I don't now 100% if this is correct
run_probe(physics_latent, all_labels, "Physics --> CLASS (want High)")
run_probe(physics_latent, all_sectors, "Physics --> SECTOR (want around chance)")
run_probe(instrument_latent, all_sectors, "Instrument --> SECTOR (want High)")
run_probe(instrument_latent, all_labels, "Instrument --> CLASS (want around chance)")


