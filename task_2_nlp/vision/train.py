"""Train a ResNet model on the Animals10 dataset (improved)."""

import os
from pathlib import Path
import time

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# training parameters
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data/animals"
MODEL_DIR = BASE_DIR / "../models/vision/"
MODEL_PATH = MODEL_DIR / "model_resnet_animals.pth"

BATCH_SIZE = 32
EPOCHS = 10
LR_FC = 1e-3  # learning rate for frozen FC
LR_FINETUNE = 1e-4  # learning rate for full fine-tuning
NUM_CLASSES = 10
PATIENCE = 3  # early stopping patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data transforms & augmentation
# -----------------------------
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# -----------------------------
# Dataset & DataLoader
# -----------------------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(
    datasets.ImageFolder(DATA_DIR, transform=transform_val), val_idx
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model setup
# -----------------------------
model = models.resnet18(pretrained=True)

# Freeze convolutional base
for param in model.parameters():
    param.requires_grad = False

# Replace FC layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_FC)

scaler = torch.cuda.amp.GradScaler()

# -----------------------------
# Training with early stopping
# -----------------------------
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0, 0
    start_time = time.time()

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
        f"Time: {time.time()-start_time:.1f}s"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Model saved to {MODEL_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
