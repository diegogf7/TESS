import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from src.data.data import ClassificationDataset
from src.models.s4d import S4Classifier

BATCH_SIZE = 256
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet"
VAL_PATH  = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_val_30min.parquet"
MS16_CKPT = "/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth"

FINETUNE_INIT = os.environ.get("FINETUNE_INIT", "jepa")
FINETUNE_CHECKPOINT = os.environ.get("FINETUNE_CKPT", "/orcd/scratch/orcd/006/diegogon/checkpoints/ft_jepa_init.pth")

dataset = ClassificationDataset(DATA_PATH, grid_length = 1024)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

validation_dataset = ClassificationDataset(VAL_PATH, grid_length = 1024)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

model = S4Classifier(d_input= 1, d_model = 256, n_layers = 4, dropout = 0.2, n_tokens = 16, token_dim = 16, n_classes = 8, readout = "mean_std").to(DEVICE)

if FINETUNE_INIT == "jepa":

    checkpoint = torch.load(MS16_CKPT, map_location = DEVICE)
    prefix = "target_encoder."
    backbone_sd = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict = True)

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3}
], weight_decay = 0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)
criteria = nn.CrossEntropyLoss()

best_value = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    all_predictions = []
    all_labels = []

    for batch_idx, (flux, mask, labels) in enumerate(dataloader):
        flux = flux.to(DEVICE)
        flux = flux.unsqueeze(-1)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(flux, mask = mask)
        loss = criteria(outputs, labels)

        loss.backward()
        optimizer.step()
        
        total_loss = total_loss + loss.item()

        predictions = outputs.argmax(dim =1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 ==0:
            print(f"Epoch {epoch + 1} / {EPOCHS}, batch {batch_idx}, loss: {loss.item():.5f}")

    average_loss = total_loss / len(dataloader)

    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    print(f"Epoch {epoch + 1} / {EPOCHS}, train average loss: {average_loss:.5f}, balanced accuracy: {balanced_accuracy: .5f}")

    scheduler.step()

    model.eval()
    val_total_loss = 0

    val_all_predictions = []
    val_all_labels = []

    with torch.no_grad():
        for batch_idx, (flux, mask, labels) in enumerate(validation_dataloader):
            flux = flux.to(DEVICE)
            flux = flux.unsqueeze(-1)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(flux, mask = mask)
            loss = criteria(outputs, labels)
            val_total_loss = val_total_loss + loss.item()

            predictions = outputs.argmax(dim = 1)
            val_all_predictions.extend(predictions.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())

    
    average_loss_validation = val_total_loss / len(validation_dataloader)
    validation_balanced_accuracy = balanced_accuracy_score(val_all_labels, val_all_predictions)
    print(f"Epoch {epoch + 1} / {EPOCHS}, VAL average loss: {average_loss_validation:.5f}, balanced accuracy: {validation_balanced_accuracy:.5f}")

    if validation_balanced_accuracy > best_value:

        best_value = validation_balanced_accuracy
        torch.save(model.state_dict(), FINETUNE_CHECKPOINT)
        print(f"The saved new best {validation_balanced_accuracy:.5f}")

print(f"Best val balanced accuracy ({FINETUNE_INIT} init): {best_value:.5f}")
        
