#this is all claude code
# Fine-tune arm of the instrument benchmark. Identical data, split, infill,
# loss, and val protocol to train_instrument_ceiling.py -- the ONLY difference
# is the encoder is initialized from a pretrained JEPA checkpoint (JEPA_INIT)
# instead of random. Compare best val bal-acc against the ceiling run.

import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import balanced_accuracy_score

from src.data.data import normalize, resample_to_grid
from src.loss_function.gap_infill import infill_gaps
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

SECTOR = int(os.environ["TESS_SECTOR"])
TARGET = os.environ["TESS_TARGET"]
assert TARGET in ("camera", "camccd", "area"), f"bad TESS_TARGET {TARGET!r}"

JEPA_INIT = os.environ["JEPA_INIT"]  # path to a pretrained gapblind JEPA checkpoint
LR = float(os.environ.get("JEPA_LR", "1e-3"))

DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_full_area.parquet")
SEED = int(os.environ.get("JEPA_SEED", "0"))
CHECKPOINT = os.environ.get("JEPA_CKPT",
                            f"/orcd/scratch/orcd/006/diegogon/checkpoints/finetune_{TARGET}_sec{SECTOR}_s{SEED}.pth")
EPOCHS = int(os.environ.get("JEPA_EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("JEPA_BATCH", "128"))
GRID_LENGTH = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------- data: one sector, star-disjoint split ----------------
df = pd.read_parquet(DATA_PATH, columns=["GAIADR3", "sector", "camera", "ccd", "area", "time", "flux"])
df = df[df["sector"] == SECTOR].reset_index(drop=True)
if len(df) == 0:
    raise SystemExit(f"no rows for sector {SECTOR} in {DATA_PATH}")

if TARGET == "camera":
    raw_labels = df["camera"].to_numpy()
elif TARGET == "camccd":
    raw_labels = (df["camera"] * 10 + df["ccd"]).to_numpy()
else:
    raw_labels = df["area"].to_numpy()

classes = np.sort(np.unique(raw_labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
labels = np.array([class_to_idx[c] for c in raw_labels])
n_classes = len(classes)

stars = df["GAIADR3"].to_numpy()
unique_stars = np.random.default_rng(SEED).permutation(np.unique(stars))
n_val_stars = max(1, int(0.2 * len(unique_stars)))
val_stars = set(unique_stars[:n_val_stars].tolist())
is_val = np.array([s in val_stars for s in stars])

print(f"git commit: {git_commit()}")
print(f"config: sector={SECTOR} target={TARGET} ({n_classes} classes, "
      f"chance {1.0 / n_classes:.4f}) seed={SEED} epochs={EPOCHS} lr={LR} -> {CHECKPOINT}")
print(f"init: {JEPA_INIT}")
print(f"data: {DATA_PATH}")
print(f"{len(df)} curves, {len(unique_stars)} stars -> "
      f"{int((~is_val).sum())} train / {int(is_val.sum())} val curves (star-disjoint)")
counts = np.bincount(labels, minlength=n_classes)
print(f"class counts: min {counts.min()} / max {counts.max()}")


class CeilingDataset(Dataset):
    def __init__(self, frame, frame_labels):
        self.frame = frame.reset_index(drop=True)
        self.labels = frame_labels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        flux = normalize(np.array(row["flux"]))
        grid_flux, observed = resample_to_grid(np.array(row["time"]), flux, GRID_LENGTH)
        return (torch.tensor(grid_flux, dtype=torch.float32),
                torch.tensor(observed, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


train_ds = CeilingDataset(df[~is_val], labels[~is_val])
val_ds = CeilingDataset(df[is_val], labels[is_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(SEED))

# ---------------- model: pretrained JEPA encoder + linear head ----------------
jepa = build_instrument_jepa()
jepa.load_state_dict(torch.load(JEPA_INIT, map_location="cpu"))
encoder = jepa.target_encoder  # the EMA encoder all the probe numbers read
for p in encoder.parameters():
    p.requires_grad = True     # was frozen inside the JEPA


class FineTuneClassifier(nn.Module):
    def __init__(self, encoder, n_classes, n_tokens=16, token_dim=16):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(n_tokens * token_dim, n_classes)

    def forward(self, flux, mask):
        z = self.encoder(flux, mask)  # (B, 16, 16); input already (B, L, 1)
        return self.head(z.reshape(z.shape[0], -1))


model = FineTuneClassifier(encoder, n_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = -1.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for flux, mask, y in train_loader:
        flux, mask, y = flux.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        filled, _ = infill_gaps(flux, mask)      # gap-blind: infilled, no mask
        optimizer.zero_grad()
        logits = model(filled.unsqueeze(-1), None)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(999)                   # fixed val infills -> stable metric
        for flux, mask, y in val_loader:
            flux, mask = flux.to(DEVICE), mask.to(DEVICE)
            filled, _ = infill_gaps(flux, mask)
            logits = model(filled.unsqueeze(-1), None)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(y.numpy())
    val_bacc = balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))
    marker = ""
    if val_bacc > best_val:
        best_val = val_bacc
        torch.save(model.state_dict(), CHECKPOINT)
        marker = "  <- saved best"
    print(f"epoch {epoch + 1}/{EPOCHS}  train loss {total_loss / len(train_loader):.4f}  "
          f"val bal-acc {val_bacc:.4f}{marker}")

print(f"BEST val balanced accuracy: {best_val:.4f} "
      f"(sector {SECTOR}, target {TARGET}, {n_classes} classes, chance {1.0 / n_classes:.4f}, "
      f"lr {LR}, init {JEPA_INIT})")
