import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns


from data import ClassificationDataset

from s4d import S4Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['APERIODIC', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE','GDOR_SPB', 'INSTRUMENT/JUNK', 'RRLYR_CEPH', 'SOLARLIKE']

model = S4Model(d_input = 1, d_output = 8, d_model = 256, n_layers = 4, dropout = 0.2).to(DEVICE)
model.load_state_dict(torch.load('/orcd/scratch/orcd/006/diegogon/checkpoints/s4d_classification.pth', map_location=DEVICE))

model.eval()
val_dataset = ClassificationDataset('/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_test.parquet')

val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)


all_predictions = []
all_labels = []
with torch.no_grad():
    for flux, mask, labels in val_loader:
        flux = flux.to(DEVICE)
        flux = flux.unsqueeze(-1)

        mask = mask.to(DEVICE)

        outputs = model(flux, mask = mask)
        predictions = outputs.argmax(dim = 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())


confusion_matrix_view = confusion_matrix(all_labels, all_predictions)
per_class = confusion_matrix_view.diagonal() / confusion_matrix_view.sum(axis=1)

confusion_matrix_normalized = confusion_matrix_view / confusion_matrix_view.sum(axis = 1, keepdims = True) * 100

plt.figure(figsize = (10,8))
sns.heatmap(confusion_matrix_normalized, annot = True, fmt = '.1f', cmap = 'Greens', xticklabels = CLASSES, yticklabels = CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("S4D Confusion Matrix on TESS Stellar variability classes")
plt.xticks(rotation = 45, ha = "right")
plt.yticks(rotation = 0)
plt.tight_layout()
plt.savefig('/orcd/scratch/orcd/006/diegogon/plots/s4d_confusion_matrix.png')
