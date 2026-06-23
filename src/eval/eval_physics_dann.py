import torch
import numpy as np

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.models.physics_jepa import PhysicsJEPA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
CHECKPOINT = "/orcd/scratch/orcd/006/diegogon/checkpoints/training_physics_dann.pth"
BATCH_SIZE = 256


def run_probe(X, y, name):
    uni, counts = np.unique(y, return_counts=True)
    keep = uni[counts >= 2]
    m = np.isin(y, keep)
    X, y = X[m], y[m]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    probe = LogisticRegression(max_iter=5000)
    probe.fit(x_train, y_train)
    predictions = probe.predict(x_test)

    accuracy = balanced_accuracy_score(y_test, predictions)
    chance = 1 / len(np.unique(y))
    print(f"{name}: balanced accuracy = {accuracy:.6f} (chance = {chance:.4f})")


dataset = DualEvalDataset(DATA_PATH, 1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=4)

# n_sectors must match what training used so the checkpoint loads cleanly
n_sectors = dataset.df["sector"].nunique()
print(f"{n_sectors} sectors")

model = PhysicsJEPA(d_model=256, n_layers=4, dropout=0.2, momentum=0.996, n_sectors=n_sectors, lambd=1.0).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

physics_latent = []
all_sectors = []
all_labels = []

with torch.no_grad():
    for flux, mask, sector, label in loader:
        flux = flux.to(DEVICE).unsqueeze(-1)
        mask = mask.to(DEVICE)

        z = model.target_encoder(flux, mask)
        physics_latent.append(z.reshape(z.shape[0], -1).cpu().numpy())

        all_sectors.append(sector.numpy())
        all_labels.append(label.numpy())

physics_latent = np.concatenate(physics_latent)
all_sectors = np.concatenate(all_sectors)
all_labels = np.concatenate(all_labels)

print(f"Mean std physics latent {physics_latent.std(0).mean()}")

run_probe(physics_latent, all_labels,  "latent --> CLASS  (want high)")
run_probe(physics_latent, all_sectors, "latent --> SECTOR (want ~chance)")
