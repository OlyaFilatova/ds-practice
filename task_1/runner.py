from utils.logs import store_logs
from utils.metrics.accuracy import accuracy
from wrapper.mnist_classifier import Algorithm, MnistClassifier

def run(algorithm: Algorithm, training, test):
    print("creating classifier")
    classifier = MnistClassifier(algorithm)

    print('training')
    classifier.train(training[0], training[1])

    print('predicting')
    res = classifier.predict(test[0])

    print('result')
    print(res)

    metric_accuracy, expected_n_predicted = accuracy(test[1], [item['prediction'] for item in res])

    print('Expected vs. Predicted')
    print(expected_n_predicted)

    print(f'Accuracy {metric_accuracy}%')

    logs = {
        "algorithm": algorithm,
        # "note": "", # Put a note here if there are any changes to the algorithm or hyperparams.
        # Alternative approach would be saving each experiment as a commit. This way we can track how code changes affect models' performance.
        "metrics": {
            "accuracy": metric_accuracy
        },
        "samples": [{
            "correct": int(item["prediction"]) == int(test[1][idx]),
            "prediction": item["prediction"],
            "expected": test[1][idx],
            "confidence": item["confidence"],
            "image_index": test[2][idx]
        } for idx, item in enumerate(res)],
        "training_set_indices": training[2],
    }

    store_logs(logs)
