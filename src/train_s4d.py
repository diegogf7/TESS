import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from data import LightCurveDataset, ClassificationDataset
from s4d import S4Model

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val.parquet"


dataset = ClassificationDataset(DATA_PATH, grid_length = 1024)
val_dataset = ClassificationDataset(VAL_PATH, grid_length = 1024)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)

model = S4Model(d_input = 1, d_output = 8, d_model = 256, n_layers = 4, dropout = 0.2).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

criteria = nn.CrossEntropyLOSS()