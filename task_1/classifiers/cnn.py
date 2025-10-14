"""Convolutional Neural Network classifier for MNIST-like data."""
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from classifiers.base import MnistClassifierInterface
from utils.response import format_response

class CNNModel(nn.Module):
    """Convolutional Neural Network Model for MNIST-like data."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Define the forward pass of the CNN model."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CnnMnistClassifier(MnistClassifierInterface):
    """
    Convolutional Neural Network classifier for MNIST-like data.

    Args:
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 3.
        device (str, optional): Device to use for training (e.g., 'cuda' or 'cpu').
            Defaults to None.

    Example:
        Input:
            images = [array of shape (N, 28, 28)]
            labels = [array of shape (N,)]
        Output:
            Trains the CNN model and returns predictions for test images.
    """
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
        y_tensor = torch.tensor(
            np.array(labels).astype(np.int64),
            dtype=torch.long,
            device=self.device
            ) if labels is not None else None
        return x_tensor, y_tensor

    def train(self, x_train, y_train):
        """
        Train the CNN model on the provided dataset.

        Args:
            x_train (list or np.ndarray): List of training images.
            y_train (list or np.ndarray): Corresponding labels for the training images.

        Example:
            Input:
                x_train = [array of shape (N, 28, 28)]
                y_train = [array of shape (N,)]
            Output:
                Trains the model for the specified number of epochs.
        """
        self.model.train()
        for _ in range(self.epochs):
            x_tensor, y_tensor = self._prepare_tensor(x_train, y_train)
            self.optimizer.zero_grad()
            output = self.model(x_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, x_test):
        """
        Predict MNIST digits and confidence for each image.

        x_test: flattened array

        Returns: {predictions, confidences}
        """
        self.model.eval()
        x_tensor, _ = self._prepare_tensor(x_test)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

        return format_response(preds.cpu().numpy(), confidences.cpu().numpy())
