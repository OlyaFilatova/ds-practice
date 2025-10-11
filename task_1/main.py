from utils.loader import load_mnist
from wrapper.mnist_classifier import Algorithm
from runner import run

def main(algorithm: Algorithm, train_size=30000, test_size=10):

    print("loading dataset")
    training, test = load_mnist(train_size=train_size, test_size=test_size)

    run(algorithm, training, test)


if __name__ == "__main__":
    main(Algorithm.CNN.value)
