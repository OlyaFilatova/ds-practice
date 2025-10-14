"""Feed-Forward Neural Network classifier for MNIST-like data."""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

from classifiers.base import MnistClassifierInterface
from utils.response import format_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MNISTFFN(nn.Module):
    """Feed-Forward Neural Network Model for MNIST-like data."""
    def __init__(self, input_dim=28*28, hidden_dims=None, num_classes=10):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Define the forward pass of the FNN model."""
        x = x.view(x.size(0), -1)  # Flatten 28x28 -> 784
        return self.net(x)


class FnnMnistClassifier(MnistClassifierInterface):
    """
    Feed-Forward Neural Network classifier for MNIST-like data.

    Args:
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        device (str, optional): Device to use for training (e.g., 'cuda' or 'cpu').
            Defaults to None.
    """
    def __init__(self, lr=1e-3, epochs=5, batch_size=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTFFN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _prepare_tensor(self, images, labels=None):
        """Convert Python lists or numpy arrays to Torch tensors."""
        if isinstance(images, list):
            images = np.stack(images)
        images = np.expand_dims(images, 1)  # (N, 1, 28, 28)
        x_tensor = torch.tensor(images, dtype=torch.float32) / 255.0
        y_tensor = torch.tensor(
            np.array(labels).astype(np.int64),
            dtype=torch.long
        ) if labels is not None else None
        return x_tensor, y_tensor

    def train(self, x_train, y_train):
        """
        Train the FNN model on the provided dataset.

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
        logging.info("Training Feed-Forward NN with %d samples", len(x_train))
        dataset = TensorDataset(*self._prepare_tensor(x_train, y_train))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss, correct = 0, 0
            for xb, yb in dataloader:
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
            logging.info(
                "Epoch %d/%d | Loss: %.4f | Acc: %.4f",
                epoch+1,
                self.epochs,
                avg_loss,
                acc
            )

    def predict(self, x_test):
        """
        Predict MNIST digits and confidence for each image.

        x_test: flattened array

        Returns: {predictions, confidences}
        """
        logging.info("Predicting with Feed-Forward NN on %d samples", len(x_test))
        x_tensor, _ = self._prepare_tensor(x_test)
        loader = DataLoader(x_tensor, batch_size=self.batch_size)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = F.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)

        return format_response(preds.cpu().numpy(), confidences.cpu().numpy())
