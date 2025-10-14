"""Random Forest Classifier for MNIST-like data."""
import logging
from sklearn.ensemble import RandomForestClassifier

from classifiers.base import MnistClassifierInterface
from utils.response import format_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest Classifier for MNIST-like data.

    Args:
        n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
        max_depth (int, optional): Maximum depth of the trees. Defaults to 5.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, x_train, y_train):
        """
        Train the Random Forest model on the provided dataset.

        Args:
            x_train (list or np.ndarray): Flattened training images of shape (N, 784).
            y_train (list or np.ndarray): Corresponding labels for the training images.

        Example:
            Input:
                x_train = [array of shape (N, 784)]
                y_train = [array of shape (N,)]
            Output:
                Trains the model on the provided dataset.
        """
        logging.info("Training Random Forest with %d samples", len(x_train))
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Predict MNIST digits and confidence for each image.

        x_test: flattened array

        Returns: {predictions, confidences}
        """
        logging.info("Predicting with Random Forest on %d samples", len(x_test))
        probs = self.model.predict_proba(x_test)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        return format_response(preds, confs)
