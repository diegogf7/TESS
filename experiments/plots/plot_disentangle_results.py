import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from src.data.data import ClassificationDataset
from src.models.disentangle import DisentanglementModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['APERIODIC', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE','GDOR_SPB', 'INSTRUMENT/JUNK', 'RRLYR_CEPH', 'SOLARLIKE']

BATCH_SIZE = 256
TRAIN_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"
TEST_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_test.parquet"

model = DisentanglementModel().to(DEVICE)
model.load_state_dict(torch.load('/orcd/scratch/orcd/006/diegogon/checkpoints/disentangle_combined.pth', map_location=DEVICE))
model.eval()

train_dataset = ClassificationDataset(TRAIN_PATH, grid_length = 1024)
test_dataset = ClassificationDataset(TEST_PATH, grid_length = 1024)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)


def extract_latents(loader):
    latents = []
    all_labels = []
    with torch.no_grad():
        for flux, mask, labels in loader:
            flux = flux.to(DEVICE).unsqueeze(-1)
            mask = mask.to(DEVICE)

            z_physics = model.physics_encoder(flux, mask)
            z_physics = z_physics.reshape(z_physics.shape[0], -1) #reducing one dimension


            latents.append(z_physics.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(latents), np.concatenate(all_labels)


train_latents, train_labels = extract_latents(train_loader)
test_latents, test_labels = extract_latents(test_loader)

scaler = StandardScaler()
train_latents = scaler.fit_transform(train_latents)
test_latents = scaler.transform(test_latents)


classifier = LogisticRegression(max_iter = 1000)
classifier.fit(train_latents, train_labels)
predictions = classifier.predict(test_latents)
print(f"Balanced accuracy score: {balanced_accuracy_score(test_labels, predictions)}")

confusion_matrix_view = confusion_matrix(test_labels, predictions)
confusion_matrix_normalized = confusion_matrix_view / confusion_matrix_view.sum(axis = 1, keepdims = True) * 100

plt.figure(figsize = (10,8))
sns.heatmap(confusion_matrix_normalized, annot = True, fmt = '.1f', cmap = 'Greens', xticklabels = CLASSES, yticklabels = CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Dual Encoder (physics latents) Confusion Matrix on TESS Stellar variability classes")
plt.xticks(rotation = 45, ha = "right")
plt.yticks(rotation = 0)
plt.tight_layout()
plt.savefig('/orcd/scratch/orcd/006/diegogon/plots/disentangle_confusion_matrix.png')
