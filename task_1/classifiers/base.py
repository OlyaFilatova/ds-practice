from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """Classifier interface for MNIST-like data."""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train model on MNIST images and labels.

        images: flattened array

        labels: array
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict MNIST digits and confidence for each image.

        images: flattened array
        
        Returns: {predictions, confidences}
        """
        pass
