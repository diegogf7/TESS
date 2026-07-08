import os

import matplotlib
matplotlib.use("Agg")  # headless: cluster nodes have no display
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import DualEvalDataset
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
CHECKPOINT = os.environ.get("JEPA_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth")
OUT_DIR = os.path.dirname(CHECKPOINT)
BATCH_SIZE = 256

GRID = 1024
STEPS = 50
INTEGRATED_CURVES = 200
N_SHOWS = 6

dataset = DualEvalDataset(DATA_PATH, GRID)
loader = DataLoader(dataset, BATCH_SIZE, shuffle = False, num_workers = 2)

model = build_instrument_jepa().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location = DEVICE))
model.eval()

random_number = np.random.default_rng(0)

latents = []
all_sectors = []
all_flux = []
all_mask = []

with torch.no_grad():

    for i, (flux, mask, sector, label) in enumerate(loader):

        flux = flux.to(DEVICE)
        mask = mask.to(DEVICE)
        z = model.encode(flux, mask)
        latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
        all_sectors.append(sector.numpy())
        all_flux.append(flux.cpu().numpy())
        all_mask.append(mask.cpu().numpy())

        if i % 10 == 0:
            print(f" extracting batch {i} / {len(loader)}")

all_latents = np.concatenate(latents)
sectors = np.concatenate(all_sectors)
fluxes = np.concatenate(all_flux)
masks = np.concatenate(all_mask)

unique, counts = np.unique(sectors, return_counts = True)
keep = np.isin(sectors, unique[counts >= 5])
all_latents = all_latents[keep]
sectors = sectors[keep]
fluxes = fluxes[keep]
masks = masks[keep]

index = np.arange(len(all_latents))
index_train, index_test = train_test_split(index, test_size = 0.2, stratify = sectors, random_state = 0)

scaler = StandardScaler().fit(all_latents[index_train])
probe = LogisticRegression(max_iter = 2000, class_weight = "balanced").fit(scaler.transform(all_latents[index_train]), sectors[index_train])

test_predictions = probe.predict(scaler.transform(all_latents[index_test]))
accuracy = balanced_accuracy_score(sectors[index_test], test_predictions)

print(f"Balanced probe accuracy: {accuracy:.5f} (around 80 percent needed)")

correct_index = index_test[test_predictions == sectors[index_test]]
ig_index = random_number.choice(correct_index, size = min(INTEGRATED_CURVES, len(correct_index)), replace = False)

#now I need to probe torch

W = torch.tensor(probe.coef_, dtype = torch.float32, device = DEVICE)
b = torch.tensor(probe.intercept_, dtype = torch.float32, device = DEVICE)
mu = torch.tensor(scaler.mean_, dtype = torch.float32, device = DEVICE)
sd = torch.tensor(scaler.scale_, dtype = torch.float32, device = DEVICE)

def sector_logit(flux_batch, mask_batch, column):

    a = model.target_encoder(flux_batch.unsqueeze(-1), mask_batch)

    a = a.reshape(a.shape[0], -1)
    a = (a - mu)
    a = a / sd

    weights = W[column]
    bias_class = b[column]

    score = (a * weights).sum(dim = 1) + bias_class

    return score

    

alphas = torch.linspace(0.0, 1.0, STEPS, device = DEVICE).view(-1, 1)

attributions = np.zeros((len(ig_index), GRID))
complete_gaps = np.zeros(len(ig_index))


score_diffs = np.zeros(len(ig_index))
for k, di in enumerate(ig_index):

    flux_t = torch.from_numpy(fluxes[di]).to(DEVICE)
    mask_t = torch.from_numpy(masks[di]).to(DEVICE)

    true_column = int(np.where(probe.classes_ == sectors[di])[0][0])

    path = (alphas * flux_t.unsqueeze(0)).detach().requires_grad_(True)
    logits = sector_logit(path, mask_t.unsqueeze(0).repeat(STEPS, 1), true_column)

    logits.sum().backward()

    ig = (flux_t * path.grad.mean(dim = 0)).detach().cpu().numpy()

    attributions[k] = ig

    with torch.no_grad():

        s_x = sector_logit(flux_t.unsqueeze(0), mask_t.unsqueeze(0), true_column).item()
        s_0 = sector_logit(torch.zeros_like(flux_t).unsqueeze(0), mask_t.unsqueeze(0), true_column).item()
    
    complete_gaps[k] = ig.sum() - (s_x - s_0)

    score_diffs[k] = s_x - s_0

    if k % 20 == 0:
        print(f" Integrated gradient curve {k} / {len(ig_index)}")


print(f"mean completeness gap = {np.abs(complete_gaps).mean():.5f}")
print(f"mean score(x) - score(0) = {np.abs(score_diffs).mean():.5f}")



for k in range(min(N_SHOWS, len(ig_index))):

    
    di = ig_index[k]

    figure, (ax0, ax1) = plt.subplots(2, 1, figsize = (10, 5), sharex = True, gridspec_kw = {"height_ratios": [2, 1]})

    ax0.plot(fluxes[di], ".", ms = 3)
    ax0.set_ylabel("normalized flux")

    ax0.set_title(f"sector {sectors[di]} score(x) - score(0) = {score_diffs[k]:.2f}", fontsize = 10)

    ax1.fill_between(np.arange(GRID), attributions[k], 0.0, color = "blue", alpha = 0.4)
    ax1.set_ylabel("Integrated gradient attribution")
    ax1.set_xlabel("grid index")

    figure.tight_layout()
    figure.savefig(os.path.join(OUT_DIR, f"ig_curve_{k}.png"), dpi = 150)

    plt.close(figure)



integrated_sectors = sectors[ig_index]

sector_unique = np.unique(integrated_sectors)

for s in sector_unique:
    m = integrated_sectors == s

    print(f"sector {s}: n = {m.sum():3d} mean attribute = {np.abs(attributions[m]).mean():.5f} mean score difference = {score_diffs[m].mean():.3f}")

mean_maps = np.stack([np.abs(attributions)[integrated_sectors == s].mean(axis = 0) for s in sector_unique])

figure, axis =plt.subplots(figsize = (10, 0.35 * len(sector_unique) + 2))

vmax = np.percentile(mean_maps, 99)

importa = axis.imshow(mean_maps, aspect = "auto", cmap = "viridis", vmin = 0, vmax = vmax, extent = (0, GRID, len(sector_unique) - 0.5, -0.5))

axis.set_yticks(range(len(sector_unique)))
axis.set_yticklabels(sector_unique, fontsize = 5)
axis.set_xlabel("grid index")
axis.set_ylabel("sector")

axis.set_title("mean integrated per sector")
figure.colorbar(importa, ax = axis, fraction = 0.025)
figure.savefig(os.path.join(OUT_DIR, "integrated_gradient_heatmap.png"), dpi = 150)

