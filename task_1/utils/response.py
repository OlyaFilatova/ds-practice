"""Utility functions for formatting model responses."""
from typing import TypedDict

class Response(TypedDict):
    prediction: int
    confidence: float

def format_response(predictions: list[int], confidences: list[float]) -> list[Response]:
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
