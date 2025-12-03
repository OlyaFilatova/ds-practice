"""Convolutional Neural Network classifier for MNIST-like data."""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

from classifiers.base import MnistClassifierInterface
from utils.response import format_response

# Augmentation transform for training
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
    ]
)


class CNNModel(nn.Module):
    """Convolutional Neural Network Model for MNIST-like data."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Define the forward pass of the CNN model."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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

    def __init__(self, lr=0.001, epochs=10, device=None, class_weights=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CNNModel().to(self.device)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if class_weights is not None:
            weights_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # MNIST mean/std normalization
        self.mean = 0.1307
        self.std = 0.3081

    def _prepare_tensor(self, images, labels=None, augment=False):
        """Convert lists/arrays to tensors, normalize, optionally apply augmentation."""
        if isinstance(images, list):
            images = np.stack(images)
        images = images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        if augment:
            # Apply augmentation per image
            augmented = []
            for img in images:
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(
                    0
                )
                img_aug = train_transform(img_tensor)
                augmented.append(img_aug.squeeze(0).numpy())
            images = np.stack(augmented)
        # Standardize
        images = (images - self.mean) / self.std
        x_tensor = torch.tensor(
            images, dtype=torch.float32, device=self.device
        )
        y_tensor = None
        if labels is not None:
            y_tensor = torch.tensor(
                np.array(labels).astype(np.int64),
                dtype=torch.long,
                device=self.device,
            )
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
        for _ in range(self.epochs):
            self.model.train()
            x_tensor, y_tensor = self._prepare_tensor(
                x_train, y_train, augment=True
            )
            self.optimizer.zero_grad()
            logits = self.model(x_tensor)
            loss = self.criterion(logits, y_tensor)
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
