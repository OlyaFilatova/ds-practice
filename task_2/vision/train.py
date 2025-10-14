"""Train a ResNet model on the Animals10 dataset."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm

from .preprocessing import transform

# training parameters
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data/animals"
MODEL_PATH = BASE_DIR / "../models/vision/model_resnet_animals.pth"

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
NUM_CLASSES = 10


# dataset & dataloader
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model setup
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training loop
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
