"""Base classifier interface for MNIST-like data."""

from abc import ABC, abstractmethod

from utils.response import PredictResponse


class MnistClassifierInterface(ABC):
    """Classifier interface for MNIST-like data."""

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train model on MNIST images and labels.

        x_train: flattened array

        y_train: array
        """

    @abstractmethod
    def predict(self, x_test) -> list[PredictResponse]:
        """
        Predict MNIST digits and confidence for each image.

        x_test: flattened array

        Returns: {predictions, confidences}
        """
