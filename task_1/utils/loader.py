from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(train_size = 5000, test_size=10000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size
    )

    return ((X_train, y_train), (X_test, y_test))
