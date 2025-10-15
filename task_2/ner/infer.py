"""Infer animal entities from text using a trained NER model."""
from pathlib import Path
import logging
import json
import re

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_as_json(message, **kwargs):
    """Log messages in JSON format."""
    logging.info(json.dumps({"message": message, **kwargs}))

BASE_DIR = Path(__file__).resolve().parent

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / "../models/ner")
model = AutoModelForTokenClassification.from_pretrained(BASE_DIR / "../models/ner")

# Aggregated NER pipeline
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# List of known animals for template matching
ANIMAL_LIST = [
    "cat", "dog", "cow", "horse", "sheep", "elephant",
    "butterfly", "chicken", "spider", "squirrel"
]

def tokenize_with_punct(text: str):
    """Tokenize text while keeping punctuation separate."""
    return re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text, re.UNICODE)

def extract_animals(text: str):
    """
    Extract animal entities from text using NER + template fallback.
    Extract animal entities from the given text.

    Args:
        text (str): Input text to analyze.

    Returns:
        tokens: list of str
        results: list of dict {"word": str, "index": int}
    """
    log_as_json("Extracting animals", text=text)
    tokens = tokenize_with_punct(text)
    results = []

    # --- Step 1: NER model predictions ---
    ner_entities = ner_pipeline(text)
    for ent in ner_entities:
        word = ent["word"].lower()
        if ent["entity_group"] == "ANIMAL":
            # find token index
            try:
                idx = next(i for i, t in enumerate(tokens) if t.lower() == word)
            except StopIteration:
                idx = -1
            if idx >= 0:
                results.append({"word": word, "index": idx})

    # --- Step 2: Template matching fallback for short 1-3 word templates ---
    words_lower = [t.lower() for t in tokens]
    for i, w in enumerate(words_lower):
        for animal in ANIMAL_LIST:
            # Direct match if missing from NER
            if animal == w and not any(r["word"] == animal for r in results):
                results.append({"word": animal, "index": i})
            # Catch short templates like "a dog" or "not a cat"
            elif i+1 < len(words_lower):
                two_word = f"{words_lower[i]} {words_lower[i+1]}"
                if animal in two_word and not any(r["word"] == animal for r in results):
                    results.append({"word": animal, "index": i+1})
            elif i+2 < len(words_lower):
                three_word = f"{words_lower[i]} {words_lower[i+1]} {words_lower[i+2]}"
                if animal in three_word and not any(r["word"] == animal for r in results):
                    results.append({"word": animal, "index": i+2})

    log_as_json("Extracted animals", text=text, tokens=tokens, results=results)
    return tokens, results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_sentences = [
        "This is not a cow",
        "There is a cow in the picture.",
        "There is a horse in the picture.",
        "not a dog",
        "I don't think it's a sheep",
        "Look, a cat over there!",
        "dog",
        "a dog",
        "the dog",
        "not a dog"
    ]

    for s in test_sentences:
        print(extract_animals(s))
        print("-"*40)
