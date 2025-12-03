# Task 1

Explores the following classification models using the publicly available [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) dataset:

1. Random Forest
1. Feed-Forward Neural Network
1. Convolutional Neural Network

## Solution Overview

This project implements three machine learning models to classify handwritten digits from the MNIST dataset. The models include:

- **Random Forest**: A traditional ensemble learning method for classification.
- **Feed-Forward Neural Network (FNN)**: A simple neural network architecture.
- **Convolutional Neural Network (CNN)**: A deep learning model designed for image data.

The dataset is preprocessed, and the models are trained and evaluated on the MNIST dataset. The training logs and results are saved in the `logs/` directory.

## Key Components

- **Data Preprocessing**: Located in `utils/loader.py`.
- **Random Forest Model**: Implemented in `classifiers/random_forest.py`.
- **Feed-Forward Neural Network**: Implemented in `classifiers/feed_forward_nn.py`.
- **Convolutional Neural Network**: Implemented in `classifiers/cnn.py`.
- **Training and Evaluation**: Main logic in `main.py`.

## Example Usage

```python
from wrapper.mnist_classifier import MnistClassifier, Algorithm
from utils.loader import load_mnist

# Load data
(X_train, X_test, _), (y_train, y_test, _) = load_mnist()

# Initialize and train model
classifier = MnistClassifier(algorithm=Algorithm.CNN)
classifier.train(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
print(f"Predictions: {predictions}")
```

## Directory Structure

- `data/` 
  - Contains the MNIST dataset.
- `classifiers/` 
  - Contains implementations of the Random Forest, FNN, and CNN models.
- `utils/` 
  - Utility scripts for data loading and preprocessing.
- `logs/` 
  - Stores training logs and results.
- `main.py` 
  - Entry point for training and evaluating models.

## Getting Started

Follow these steps to set up and run the project:

1. **Install Python 3**: Ensure Python3 (<3.12) is installed on your system.

2. **Clone the repository**
   ```bash
   git clone https://github.com/OlyaFilatova/ds-practice.git
   cd ds-practice/task_1_classifiers
   ```

3. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Main Script**
   ```bash
   python3 main.py
   ```

The project will train the models and save the results in the `logs/` directory.

## Reproducing Training Runs with `rerun.py`

The `rerun.py` script is designed to reproduce a previous training and evaluation run using a saved log file. It performs the following steps:

1. **Load Log File**: Reads the log file specified by the user to retrieve the training and test set indices, as well as the algorithm used.
2. **Load Dataset**: Loads the MNIST dataset based on the indices stored in the log file.
3. **Re-run Training and Evaluation**: Uses the `run` function to re-execute the training and evaluation process with the same algorithm and dataset split as the original run.

### Usage

To rerun a specific log file, execute the script with the desired log file name as a command-line argument:

```bash
python rerun.py <log_filename>
```

For example:

```bash
python rerun.py 20251011_220147.json
```
