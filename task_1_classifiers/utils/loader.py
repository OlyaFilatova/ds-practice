"""Functions to load the MNIST dataset."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(train_size=5000, test_size=10000):
    """Load the MNIST dataset and split it into training and test sets."""
    x, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )

    indices = np.arange(len(x))

    x_train, x_test, y_train, y_test, train_indices, test_indices = (
        train_test_split(
            x, y, indices, train_size=train_size, test_size=test_size
        )
    )

    return ((x_train, y_train, train_indices), (x_test, y_test, test_indices))


def load_mnist_from_indices(train_indices: list[int], test_indices: list[int]):
    """Load the MNIST dataset using specific indices for training and test sets."""
    x, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )

    indexed_x = list(x)
    indexed_y = list(y)

    x_train = [indexed_x[idx] for idx in train_indices]
    y_train = [indexed_y[idx] for idx in train_indices]

    x_test = [indexed_x[idx] for idx in test_indices]
    y_test = [indexed_y[idx] for idx in test_indices]

    return ((x_train, y_train, train_indices), (x_test, y_test, test_indices))
