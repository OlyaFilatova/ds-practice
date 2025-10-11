import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

from classifiers.base import MnistClassifierInterface


class MNISTFFN(nn.Module):
    """Feed-Forward Neural Network Model for MNIST-like data."""
    def __init__(self, input_dim=28*28, hidden_dims=[128, 64], num_classes=10):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 28x28 -> 784
        return self.net(x)


class FnnMnistClassifier(MnistClassifierInterface):
    """Feed-Forward Neural Network classifier for MNIST-like data."""
    def __init__(self, lr=1e-3, epochs=5, batch_size=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTFFN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # FIXME: should not be hard-coded
        ])

    def _prepare_tensor(self, images, labels=None):
        """Convert Python lists or numpy arrays to Torch tensors."""
        if isinstance(images, list):
            images = np.stack(images)
        images = np.expand_dims(images, 1)  # (N, 1, 28, 28)
        x_tensor = torch.tensor(images, dtype=torch.float32) / 255.0
        y_tensor = torch.tensor(np.array(labels).astype(np.int64), dtype=torch.long) if labels is not None else None
        return x_tensor, y_tensor

    def train(self, images, labels):
        """
        Train FNN on MNIST images and labels.

        images: flattened array

        labels: array
        """
        x_tensor, y_tensor = self._prepare_tensor(images, labels)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss, correct = 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()

            avg_loss = total_loss / len(dataset)
            acc = correct / len(dataset)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}") # FIXME: printed format is not best for logging

    def predict(self, images):
        """
        Predict MNIST digits and confidence for each image.

        images: flattened array

        Returns: {predictions, confidences}
        """
        x_tensor, _ = self._prepare_tensor(images)
        loader = DataLoader(x_tensor, batch_size=self.batch_size)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = F.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)

        return [
            { "prediction": prediction,  "confidence": confidence } 
            for prediction, confidence in 
                list(zip(preds.cpu().numpy(), confidences.cpu().numpy()))
        ]
