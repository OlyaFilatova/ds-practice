"""Rerun training and evaluation using a saved log file."""
import argparse
import json
import logging

from utils.logs import load_log
from utils.loader import load_mnist_from_indices
from runner import run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_as_json(message, **kwargs):
    """Log a message with additional context as a JSON object."""
    logging.info(json.dumps({"message": message, **kwargs}))

def rerun(log_filename):
    """
    Rerun training and evaluation using a saved log file.

    Args:
        log_filename (str): Path to the log file containing training 
            and test indices, and the algorithm used.

    Example:
        Input:
            log_filename = "20251011_220147.json"
        Output:
            Re-runs the training and evaluation process using the data 
                and algorithm specified in the log file.
    """
    log_as_json("Loading log file", log_filename=log_filename)
    logs = load_log(log_filename)

    train_indices = logs["training_set_indices"]
    test_indices = [sample["image_index"] for sample in logs["samples"]]

    log_as_json("Loaded indices", train_indices=train_indices, test_indices=test_indices)

    # The dataset is loaded using indices from the log file
    # to ensure the exact same data split is used as in the original run.
    training, test = load_mnist_from_indices(train_indices, test_indices)

    log_as_json("Re-running with algorithm", algorithm=logs["algorithm"])
    run(logs["algorithm"], training, test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun training and evaluation using a log file.")
    parser.add_argument("log_filename", type=str, help="Path to the log file.")
    args = parser.parse_args()

    rerun(args.log_filename)
