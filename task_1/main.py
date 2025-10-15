"""Main module to run the MNIST classification task."""
import logging
from utils.loader import load_mnist
from wrapper.mnist_classifier import Algorithm
from runner import run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(algorithm: Algorithm, train_size=50000, test_size=10000):
    """
    Main function to initialize the dataset and run the specified algorithm.

    Args:
        algorithm (Algorithm): The algorithm to use for training and evaluation
            (e.g., Algorithm.CNN).
        train_size (int, optional): Number of training samples to use. Defaults to 30000.
        test_size (int, optional): Number of test samples to use. Defaults to 10.

    Example:
        Input:
            algorithm = Algorithm.CNN
            train_size = 30000
            test_size = 10
        Output:
            Trains and evaluates the CNN model on the specified dataset split.
    """
    logging.info("Loading dataset with train size: %d, test size: %d", train_size, test_size)
    training, test = load_mnist(train_size=train_size, test_size=test_size)

    logging.info("Running algorithm: %s", algorithm)
    run(algorithm, training, test)


if __name__ == "__main__":
    main(Algorithm.CNN)
