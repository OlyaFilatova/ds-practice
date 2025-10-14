"""Infer animal entities from text using a trained NER model."""
from pathlib import Path
import re
import logging
import json

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Replace logging with structured JSON logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_as_json(message, **kwargs):
    """Log messages in JSON format."""
    logging.info(json.dumps({"message": message, **kwargs}))

BASE_DIR = Path(__file__).resolve().parent

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / "../models/ner")
model = AutoModelForTokenClassification.from_pretrained(BASE_DIR / "../models/ner")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


ANIMAL_LIST = [
    "cat", "dog", "cow", "horse", "sheep", "elephant",
    "butterfly", "chicken", "spider", "squirrel"
]

def extract_animals(text: str):
    """
    Extract animal entities from the given text.

    Args:
        text (str): Input text to analyze.

    Returns:
        tuple: A tuple containing tokens (list of str) and detected animals (list of dict).

    Example:
        Input:
            text = "There is a cow in the picture."
        Output:
            (["There", "is", "a", "cow", "in", "the", "picture, "."], [{"word": "cow", "index": 3}])
    """
    log_as_json("Extracting animals", text=text)
    entities = ner_pipeline(text)
    results = []
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    for ent in entities:
        word = ent["word"].lower()
        for animal in ANIMAL_LIST:
            if animal in word:
                try:
                    idx = tokens.index(animal)
                except ValueError:
                    idx = 0
                results.append({"word": animal, "index": idx})
    log_as_json("Extracted animals", text=text, tokens=tokens, results=results)
    return tokens, results

if __name__ == "__main__":
    print(extract_animals("This is not a cow"))

    print(extract_animals("There is a cow in the picture."))
