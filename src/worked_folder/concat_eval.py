import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument_jepa import build_instrument_jepa
from src.worked_folder.latent_jepa import build_latent_jepa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
MS16_CHECKPOINT = os.environ.get("MS16_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth")
INSTRUMENT_CHECKPOINT = os.environ.get("INST_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")
BATCH_SIZE = 256
POPULATED_MIN = 50


def get_latents(model, loader):

    model.eval()
    latents = []
    sectors = []
    labels = []

    with torch.no_grad():
        for i, (flux, mask, sector, label) in enumerate(loader):
            z = model.encode(flux.to(DEVICE), mask.to(DEVICE))
            latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
            sectors.append(sector.numpy())
            labels.append(label.numpy())
            
            if i % 10 == 0:
                print(f" extracting batch {i} / {len(loader)}")

    
    return np.concatenate(latents), np.concatenate(sectors), np.concatenate(labels)


def class_probe_grouped(X, y, groups):
    unique, counts = np.unique(y, return_counts = True)

    keep = unique[counts >= 2]
    m = np.isin(y, keep)

    X = X[m]
    y = y[m]
    groups = groups[m]

    split = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
    train_index, test_index = next(split.split(X, y, groups))
    x_train = X[train_index]
    x_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    probe = KNeighborsClassifier(n_neighbors = 20)
    probe.fit(x_train, y_train)

    return balanced_accuracy_score(y_test, probe.predict(x_test))

#code just from sector probe v2 (just straight copied)
def sector_probe_v2(X, y, tag):

    unique, count = np.unique(y, return_counts = True)

    dropped = unique[count < 5]

    if len(dropped) > 0:
        print(f" {tag} dropped sectors with <5 rows: {list(dropped)}")
    keep = unique[count >= 5]

    m = np.isin(y, keep)
    X, y = X[m], y[m]

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter = 2000, class_weight = "balanced"),
    )

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    y_prediction = cross_val_predict(pipeline, X, y, cv = cv)

    sectors = np.unique(y)
    n_rows = {s: (y == s).sum() for s in sectors}

    recall = {s: (y_prediction[y == s] == s).mean() for s in sectors}

    populated = [s for s in sectors if n_rows[s] >= POPULATED_MIN]

    all_accuracy = balanced_accuracy_score(y, y_prediction)
    pop_accuracy = np.mean([recall[s] for s in populated])

    return all_accuracy, pop_accuracy


#are we making sure that it's just the certain times and the certain missions?

dataset = DualEvalDataset(DATA_PATH, 1024)

loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)
print(f"{len(dataset)} samples, {len(loader)} batches")

all_tics = dataset.df["TIC"].to_numpy()

os.environ["JEPA_NTOKENS"] = "16"
os.environ["JEPA_TOKENDIM"] = "16"

os.environ["JEPA_READOUT"] = "mean" #the instrument configuration
instrument = build_instrument_jepa().to(DEVICE)
instrument.load_state_dict(torch.load(INSTRUMENT_CHECKPOINT, map_location = DEVICE))

os.environ["JEPA_READOUT"] = "mean_std" #the physics configuration

physics = build_latent_jepa().to(DEVICE)
missing, unexpected = physics.load_state_dict(torch.load(MS16_CHECKPOINT, map_location = DEVICE), strict = False)

print("16 dim model misses:", missing, "and unexpected:", unexpected)

X_physics, y_sector, y_class = get_latents(physics, loader)
X_instrument, _, _ = get_latents(instrument, loader)

features = {
    "Physics": X_physics,
    "Instrument": X_instrument,
    "Concatenate": np.hstack([X_physics, X_instrument])
}

for name, X in features.items():
    classes = class_probe_grouped(X, y_class, all_tics)
    all_sectors, populated_sectors = sector_probe_v2(X, y_sector, name)
    print(f"{name}'s class = {classes:.5f}, sector = {all_sectors:.5f}, sector populated = {populated_sectors:.5f}")
