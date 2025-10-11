from utils.logs import load_log
from utils.loader import load_mnist_from_indices
from runner import run

def rerun(log_filename):
    logs = load_log(log_filename)

    train_indices = logs["training_set_indices"]
    
    test_indices = [sample["image_index"] for sample in logs["samples"]]

    print("loading dataset")
    training, test = load_mnist_from_indices(train_indices, test_indices)

    run(logs["algorithm"], training, test)


if __name__ == "__main__":
    rerun("20251011_220147.json")
