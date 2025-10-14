from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance using precision, recall, f1-score, and accuracy.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.

    Returns:
        dict: A dictionary containing precision, recall, f1-score, and accuracy.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }
