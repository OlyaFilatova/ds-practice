"""Infer the animal class from an image using a pre-trained ResNet model."""
from pathlib import Path
import logging
import json

from PIL import Image
import torch
from torchvision import models

from .preprocessing import transform

CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant",
               "horse", "sheep", "spider", "squirrel"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "../models/vision/model_resnet_animals.pth"

# load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Replace logging with structured JSON logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_as_json(message, **kwargs):
    """Log a message with additional context as a JSON object."""
    logging.info(json.dumps({"message": message, **kwargs}))

def classify_animal(image_path: str):
    """
    Classify the animal in the given image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Predicted animal class name.

    Example:
        Input:
            image_path = "test_images/dog/0003.jpeg"
        Output:
            "dog"
    """
    log_as_json("Classifying animal", image_path=image_path)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_tensor).argmax(1).item()
    log_as_json("Predicted class", class_name=CLASS_NAMES[pred])
    return CLASS_NAMES[pred]

if __name__ == "__main__":
    print(classify_animal("test_images/squirrel/0001.jpeg"))
