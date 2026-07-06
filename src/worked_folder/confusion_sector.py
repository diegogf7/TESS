"""
Per-sector confusion diagnostic for the instrument JEPA latent.

Same latent extraction + probe split as eval_instrument_jepa.py, but keeps the
KNN's per-sample predictions to show WHERE the sector probe loses accuracy:
concentrated in a few confusable sector pairs (=> near the physical ceiling,
stop tuning) or diffuse across many sectors (=> real headroom, keep pushing).

Outputs:
    <checkpoint dir>/sector_confusion.png   row-normalised (%) confusion matrix
plus printed per-sector recall (worst first) and the top confused pairs.

    JEPA_CKPT picks the checkpoint (default instrument_jepa.pth)
    python -m src.worked_folder.confusion_sector
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: cluster nodes have no display
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")
OUT_PNG = os.path.join(os.path.dirname(CHECKPOINT), "sector_confusion.png")
BATCH_SIZE = 256

# ---- extract latents (identical to eval_instrument_jepa.py) ----
print(f"device: {DEVICE}")
dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)
print(f"{len(dataset)} samples, {len(loader)} batches")

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

latents, all_sectors = [], []
with torch.no_grad():
    for i, (flux, mask, sector, label) in enumerate(loader):
        z = model.encode(flux.to(DEVICE), mask.to(DEVICE))
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        if i % 10 == 0:
            print(f"  extracting batch {i}/{len(loader)}")

X = np.concatenate(latents)
y = np.concatenate(all_sectors)

# ---- same filter + split + probe as run_probe (ungrouped sector branch) ----
uni, counts = np.unique(y, return_counts=True)
keep = uni[counts >= 2]
m = np.isin(y, keep)
X, y = X[m], y[m]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

probe = KNeighborsClassifier(n_neighbors=20)
probe.fit(x_train, y_train)
y_pred = probe.predict(x_test)

sectors = np.unique(y)
acc = balanced_accuracy_score(y_test, y_pred)
print(f"\nsector balanced accuracy = {acc:.4f} (chance = {1/len(sectors):.4f})  [should match eval]")

# ---- per-sector recall, worst first: these are what drag balanced accuracy ----
cm = confusion_matrix(y_test, y_pred, labels=sectors)
support = cm.sum(axis=1)
recall = np.diag(cm) / np.maximum(support, 1)

print("\nper-sector recall (worst first):")
for i in np.argsort(recall):
    print(f"  sector {sectors[i]:>3}: recall {recall[i]:.2f}  (n_test={support[i]})")

# ---- top confused pairs ----
cm_pct = cm / np.maximum(support[:, None], 1)
off = cm_pct.copy()
np.fill_diagonal(off, 0.0)
order = np.argsort(off, axis=None)[::-1]
print("\ntop confused pairs (true -> predicted, % of that sector's test samples):")
for flat_idx in order[:12]:
    i, j = np.unravel_index(flat_idx, off.shape)
    if off[i, j] < 0.05:
        break
    print(f"  {sectors[i]:>3} -> {sectors[j]:>3}: {100 * off[i, j]:.0f}%")

# ---- heatmap ----
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(100 * cm_pct, cmap="viridis", vmin=0, vmax=100)
ax.set_xticks(range(len(sectors)))
ax.set_xticklabels(sectors, rotation=90, fontsize=7)
ax.set_yticks(range(len(sectors)))
ax.set_yticklabels(sectors, fontsize=7)
ax.set_xlabel("predicted sector")
ax.set_ylabel("true sector")
ax.set_title(f"sector confusion (row-normalised %), balanced acc {acc:.3f}")
for i in range(len(sectors)):
    for j in range(len(sectors)):
        v = 100 * cm_pct[i, j]
        if v >= 10:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="black" if v >= 60 else "white")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\nsaved {OUT_PNG}")
