import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument_jepa import build_instrument_jepa
from src.worked_folder.masked_s4d import build_masked_s4d

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
JEPA_CKPT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")
MASKED_CKPT = os.environ.get("MASKED_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/masked_s4d.pth")
BATCH_SIZE = 256
POPULATED_MIN = 50

def extract_latents(model, loader):
    model.eval()
    latents = []
    sectors = []

    with torch.no_grad():

        for i, (flux, mask, sector, label) in enumerate(loader):
            z = model.encode(flux.to(DEVICE), mask.to(DEVICE))
            latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
            sectors.append(sector.numpy())

            if i % 10 == 0:
                print(f" extracting batch {i} / {len(loader)}")
    return np.concatenate(latents), np.concatenate(sectors)

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
    rare = [s for s in sectors if n_rows[s] < POPULATED_MIN]

    all_accuracy = balanced_accuracy_score(y, y_prediction)
    pop_accuracy = np.mean([recall[s] for s in populated])

    print(f"  [{tag}] all sectors ({len(sectors)}-way, chance {1/len(sectors):.3f}): "
          f"balanced acc = {all_accuracy:.4f}")
    print(f"  [{tag}] populated only ({len(populated)} sectors, >={POPULATED_MIN} rows): "
          f"balanced acc = {pop_accuracy:.4f}")
    for s in rare:
        print(f"  [{tag}]   rare sector {s}: recall {recall[s]:.2f} (n={n_rows[s]})")
    return all_accuracy, pop_accuracy




print(f"device: {DEVICE}")
dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)
print(f"{len(dataset)} samples, {len(loader)} batches")

results = {}

print("\n=== instrument JEPA ===")
jepa = build_instrument_jepa().to(DEVICE)
jepa.load_state_dict(torch.load(JEPA_CKPT, map_location=DEVICE))
X, y = extract_latents(jepa, loader)
results["instrument_jepa"] = sector_probe_v2(X, y, "jepa")

print("\n=== masked-S4D ===")
masked = build_masked_s4d().to(DEVICE)
masked.load_state_dict(torch.load(MASKED_CKPT, map_location=DEVICE))
X, y = extract_latents(masked, loader)
results["masked_s4d"] = sector_probe_v2(X, y, "masked")

print("\n=== HEAD-TO-HEAD (protocol v2: logistic regression + stratified 5-fold CV) ===")
print(f"{'model':<18} {'all sectors':>12} {'populated':>12}")
for name, (all_acc, pop_acc) in results.items():
    print(f"{name:<18} {all_acc:>12.4f} {pop_acc:>12.4f}")
