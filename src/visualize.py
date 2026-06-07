import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import normalize, resample_to_grid, LightCurveDataset
from model import masking_autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = masking_autoencoder()
model.load_state_dict(torch.load('mae_final.pth', map_location = DEVICE))
model.eval()


DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/tess_classification.parquet"
df = pd.read_parquet(DATA_PATH)
#getting the first instance of the eclipse data just to plot it for fun
eclipse = df[df['label'] == 'ECLIPSE'].iloc[0]
time = np.array(eclipse['time'])
flux = np.array(eclipse['flux'])
flux_normalized = normalize(flux)
grid_flux, observed_flux = resample_to_grid(time, flux_normalized, grid_length = 1024)

#now we need to turn our flux into a tensor
flux_tensor = torch.tensor(grid_flux, dtype = torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    predictions, shuffle = model(flux_tensor)

predictions_new = predictions.squeeze(0).cpu().numpy()
grid_flux_patches = grid_flux.reshape(64, 16)

masked_patch_indexes = shuffle[0, 16:].cpu().numpy()

#need to look this over
reconstruction = grid_flux.copy()
for idx in masked_patch_indexes:
    reconstruction[idx*16:(idx+1)*16] = predictions_new[idx]

x = np.arange(1024)

plt.figure(figsize=(14, 4))
plt.plot(x, grid_flux, label='Original', color='black', alpha=0.7)
plt.plot(x, reconstruction, label='Reconstruction', color='red', alpha=0.7)

for idx in masked_patch_indexes:
    plt.axvspan(idx*16, (idx+1)*16, alpha=0.1, color='gray')

plt.title("ECLIPSE Light Curve — MAE Reconstruction")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Flux")
plt.legend()
plt.tight_layout()
plt.savefig('/orcd/scratch/orcd/006/diegogon/plots/reconstruction.png')
print("Plot saved.", flush=True)
