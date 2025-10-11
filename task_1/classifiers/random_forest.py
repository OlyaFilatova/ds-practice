from classifiers.base import MnistClassifierInterface

from sklearn.ensemble import RandomForestClassifier


class RandomForestMnistClassifier(MnistClassifierInterface):
    """Random Forest Classifier for MNIST-like data."""
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def train(self, images, labels):
        """
        Train Random Forest on MNIST images and labels.

        images: flattened array

        labels: array
        """
        self.model.fit(images, labels)

    def predict(self, images):
        """
        Predict MNIST digits and confidence for each image.

        images: flattened array

        Returns: {predictions, confidences}
        """
        probs = self.model.predict_proba(images)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        return [
            { "prediction": prediction,  "confidence": confidence } 
            for prediction, confidence in 
                list(zip(preds, confs))
        ]
