"""Utility functions for formatting model responses."""

def format_response(predictions, confidences):
    """
    Format the predictions into a standardized response structure.

    Args:
        predictions (list of dict): List of predictions with 'prediction' and 'confidence' keys.
        confidences (list of float): List of confidence scores for each prediction.

    Returns:
        dict: Formatted response containing the predictions.
    """

    return [
        { "prediction": prediction,  "confidence": confidence }
        for prediction, confidence in
            list(zip(predictions, confidences))
    ]
