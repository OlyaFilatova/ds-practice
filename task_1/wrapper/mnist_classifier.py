import enum
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from classifiers.base import MnistClassifierInterface
from classifiers.cnn import CnnMnistClassifier
from classifiers.feed_forward_nn import FnnMnistClassifier
from classifiers.random_forest import RandomForestMnistClassifier

class Algorithm(enum.Enum):
    CNN = 'cnn'
    FNN = 'nn'
    RF = 'rf'

class MnistClassifier:
    def __init__(self, algorithm: Algorithm): # cnn, rf, and nn
        mapping: dict[Algorithm, MnistClassifierInterface] = {
            "cnn": CnnMnistClassifier,
            "nn": FnnMnistClassifier,
            "rf": RandomForestMnistClassifier
        }

        if algorithm not in mapping.keys():
            raise ValueError(f'Received unexpected algorithm name "{algorithm}". Pass one of the following: {", ".join(mapping.keys())}')
        
        model_class = mapping[algorithm]
        self.model = model_class()
        
    def train(self, X_train, y_train):
        return self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
