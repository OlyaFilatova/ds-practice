import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(train_size = 5000, test_size=10000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, train_size=train_size, test_size=test_size
    )
    
    return ((X_train, y_train, train_indices), (X_test, y_test, test_indices))

def load_mnist_from_indices(train_indices: list[int], test_indices: list[int]):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    indexed_X = list(X)
    indexed_y = list(y)

    X_train = [indexed_X[idx] for idx in train_indices]
    y_train = [indexed_y[idx] for idx in train_indices]

    X_test = [indexed_X[idx] for idx in test_indices]
    y_test = [indexed_y[idx] for idx in test_indices]

    return ((X_train, y_train, train_indices), (X_test, y_test, test_indices))
