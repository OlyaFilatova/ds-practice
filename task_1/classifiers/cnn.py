import numpy as np
from classifiers.base import MnistClassifierInterface
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNModel(nn.Module):
    """Convolutional Neural Network Model for MNIST-like data."""
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CnnMnistClassifier(MnistClassifierInterface):
    """Convolutional Neural Network classifier for MNIST-like data."""
    def __init__(self, lr=0.001, epochs=3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(self.device)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_tensor(self, images, labels=None):
        """Convert Python lists or numpy arrays to Torch tensors."""
        if isinstance(images, list):
            images = np.stack(images)
        images = images.reshape(-1, 1, 28, 28)
        x_tensor = torch.tensor(images, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.array(labels).astype(np.int64), dtype=torch.long, device=self.device) if labels is not None else None
        return x_tensor, y_tensor

    def train(self, images, labels):
        """
        Train CNN on MNIST images and labels.

        images: flattened array

        labels: array
        """
        self.model.train()
        x_tensor, y_tensor = self._prepare_tensor(images, labels)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def predict(self, images):
        """
        Predict MNIST digits and confidence for each image.

        images: flattened array

        Returns: {predictions, confidences}
        """
        self.model.eval()
        x_tensor, _ = self._prepare_tensor(images)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
        
        return [
            { "prediction": prediction,  "confidence": confidence} 
            for prediction, confidence in 
                list(zip(preds.cpu().numpy(), confidences.cpu().numpy()))
        ]
