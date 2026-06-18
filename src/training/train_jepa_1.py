import torch
import torch.nn as nn

from src.data.data import DisentanglementDataset, DataLoader
from src.models.jepa_1 import JEPA_1, jepa_loss

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"


dataset = DisentanglementDataset(DATA_PATH, grid_length =1024)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

val_dataset = DisentanglementDataset(VAL_PATH, grid_length = 1024)
dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)


model = JEPA_1(d_model = 256, n_layers =4, dropout = 0.2, momentum = 0.996).to(DEVICE)
#we're only going to train our online encoder not the EMA

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

for epoch in range(EPOCHS):

    model.train()
    total_loss= 0

    for batch_idx, (anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask) in enumerate(dataloader):
