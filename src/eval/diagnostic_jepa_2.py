# ============================================================================
# TEMPORARY DIAGNOSTIC -- safe to delete after we read the result.
#
# Question: is the STANDALONE physics JEPA healthy, or does it collapse on
#           its own (i.e. with NO shared predictor pulling on it)?
#
#   healthy   (std off the floor, sector ~ chance)  -> the shared predictor in
#             the dual model is the culprit -> fix = UN-SHARE the predictor
#   collapsed (low std, sector well above chance)   -> physics branch collapses
#             fundamentally -> fix = VARIANCE term on the physics latent
#
# Needs a training_jepa2.pth trained at the CURRENT code (multi_sector_only=True).
# ============================================================================
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.data import DualEvalDataset
from src.models.jepa_2 import JEPA_2

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/training_jepa2.pth"
BATCH_SIZE = 256


def run_probe(X, y, name):
    uni, counts = np.unique(y, return_counts=True)      # drop singletons so stratify works
    keep = uni[counts >= 2]
    m = np.isin(y, keep)
    X, y = X[m], y[m]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    probe = LogisticRegression(max_iter=5000)
    probe.fit(X_tr, y_tr)
    preds = probe.predict(X_te)

    acc = balanced_accuracy_score(y_te, preds)
    chance = 1 / len(np.unique(y))
    print(f"{name}: balanced accuracy = {acc:.4f}   (chance = {chance:.4f})")


model = JEPA_2(256, 4, 0.2, 0.996).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=4)

physics_latent, all_sectors, all_labels = [], [], []

with torch.no_grad():
    for flux, mask, sector, label in loader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z = model.target_encoder(flux, mask)            # standalone physics latent (B,16,4)
        physics_latent.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        all_labels.append(label.numpy())

physics_latent = np.concatenate(physics_latent)
all_sectors = np.concatenate(all_sectors)
all_labels  = np.concatenate(all_labels)

print(f"STANDALONE physics latent std: {physics_latent.std(0).mean():.4f}")
run_probe(physics_latent, all_labels,  "STANDALONE physics -> CLASS  (want HIGH)")
run_probe(physics_latent, all_sectors, "STANDALONE physics -> SECTOR (want ~chance)")
