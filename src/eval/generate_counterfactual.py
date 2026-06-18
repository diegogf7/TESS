import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data import ClassificationDataset
from src.models.disentangle import DisentanglementModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_test.parquet"
CKPT_PATH = "/orcd/scratch/orcd/006/diegogon/checkpoints/disentangle.pth"
OUT_PATH  = "/orcd/scratch/orcd/006/diegogon/plots/counterfactual.png"

model = DisentanglementModel().to(DEVICE)

model.load_state_dict(torch.load(CKPT_PATH, map_location = DEVICE))
model.eval()

dataset = ClassificationDataset(DATA_PATH, grid_length = 1024)
fluxA, maskA, _ = dataset[0]
fluxB, maskB, _ = dataset[1]
fluxA = fluxA.unsqueeze(0).to(DEVICE)
maskA = maskA.unsqueeze(0).to(DEVICE)

fluxB = fluxB.unsqueeze(0).to(DEVICE)
maskB = maskB.unsqueeze(0).to(DEVICE)

@torch.no_grad() #need to encode all of the physics and instrument latents
def encode(flux, mask):
    z_physics = model.physics_encoder(flux.unsqueeze(-1), mask)
    z_instrument = model.instrument_encoder(flux.unsqueeze(-1), mask)

    return z_physics, z_instrument

z_physics_A, z_instrument_A = encode(fluxA, maskA)

z_physics_B, z_instrument_B = encode(fluxB, maskB)

#need to use the velocity field and noise to make out light curve now

@torch.no_grad()
def sample(z_physics, z_instrument, steps = 100):
    context = torch.cat([z_physics, z_instrument], dim = 1)

    x = torch.randn(context.shape[0], 1024, device = DEVICE)
    for i in range(steps):
        t = torch.full((context.shape[0], 1), i / steps, device = DEVICE)
        v = model.velocity_net(x, t, context)

        x = x + (v * (1.0 / steps))
    
    return x.squeeze(0).cpu().numpy()

#need to swap grid

generate_AA = sample(z_physics_A, z_instrument_A)
generate_AB = sample(z_physics_A, z_instrument_B)

generate_BB = sample(z_physics_B, z_instrument_B)
generate_BA = sample(z_physics_B, z_instrument_A)

figure, axis = plt.subplots(3, 2, figsize = (12, 8))

axis[0,0].plot(fluxA.squeeze(0).cpu()); axis[0,0].set_title("Original A")
axis[0,1].plot(fluxB.squeeze(0).cpu()); axis[0,1].set_title("Original B")

axis[1,0].plot(generate_AA); axis[1,0].set_title("Physics A + Instrument A (to rebuild A)")
axis[1,1].plot(generate_AB); axis[1,1].set_title("Physics A + Instrument B (to build counterfactual)")
axis[2,0].plot(generate_BB); axis[2,0].set_title("Physics B + Instrument B (to rebuild B)")
axis[2,1].plot(generate_BA); axis[2,1].set_title("Physics B + Instrument A (to build counterfactual)")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi = 120)
print(f"Saved figure to {OUT_PATH}")
