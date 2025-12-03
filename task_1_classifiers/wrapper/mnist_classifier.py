"""Wrapper for different MNIST classification algorithms."""

import enum

from classifiers.base import MnistClassifierInterface
from classifiers.cnn import CnnMnistClassifier
from classifiers.feed_forward_nn import FnnMnistClassifier
from classifiers.random_forest import RandomForestMnistClassifier


class Algorithm(enum.Enum):
    """Supported classification algorithms."""

    CNN = "cnn"
    FNN = "nn"
    RF = "rf"


class MnistClassifier:
    """Wrapper class for different MNIST classification algorithms."""

    def __init__(self, algorithm: Algorithm):  # cnn, rf, and nn
        mapping: dict[Algorithm, type[MnistClassifierInterface]] = {
            Algorithm.CNN: CnnMnistClassifier,
            Algorithm.FNN: FnnMnistClassifier,
            Algorithm.RF: RandomForestMnistClassifier,
        }

        if algorithm not in mapping:
            raise ValueError(f'Received unexpected algorithm name "{algorithm}".\
                              Pass one of the following: \
                             {", ".join(str(key) for key in mapping)}')

        model_class = mapping[algorithm]
        self.model = model_class()

    def train(self, x_train, y_train):
        """Train the model with the provided training data."""
        return self.model.train(x_train, y_train)

    def predict(self, x_test):
        """Make predictions on the provided test data."""
        return self.model.predict(x_test)
