"""Runner for training and evaluating the MNIST classifier."""

import hashlib
import json

from utils.logs import CustomEncoder, store_logs
from utils.metrics.accuracy import evaluate_model
from wrapper.mnist_classifier import Algorithm, MnistClassifier


def run(algorithm: Algorithm, training, test):
    """Run the training and evaluation process."""
    print("creating classifier")
    classifier = MnistClassifier(algorithm)

    print("training")
    classifier.train(training[0], training[1])

    print("predicting")
    res = classifier.predict(test[0])

    metrics = evaluate_model(
        [int(label) for label in test[1]],
        [int(item["prediction"]) for item in res],
    )

    print("metrics")
    print(json.dumps(metrics, indent=4))

    logs = {
        "hash": hashlib.sha256(
            json.dumps(
                (tuple(training[2]), tuple(test[2])), cls=CustomEncoder
            ).encode("utf-8")
        ).hexdigest(),
        "algorithm": algorithm.value,
        "metrics": metrics,
        "samples": [
            {
                "correct": int(item["prediction"]) == int(test[1][idx]),
                "prediction": item["prediction"],
                "expected": test[1][idx],
                "confidence": item["confidence"],
                "image_index": test[2][idx],
            }
            for idx, item in enumerate(res)
        ],
        "training_set_indices": training[2],
    }

    store_logs(logs)
