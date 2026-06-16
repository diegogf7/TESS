import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ClassificationDataset
from disentangle import DisentanglementModel

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
maskB = fluxB.unsqueeze(0).to(DEVICE)

@torch.no_grad()
