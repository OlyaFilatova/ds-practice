from classifiers.base import MnistClassifierInterface

from sklearn.ensemble import RandomForestClassifier


class RandomForestMnistClassifier(MnistClassifierInterface):
    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if not self.model:
            raise RuntimeError('Call "train" method before the "predict" method.')
        return self.model.predict(X_test)
