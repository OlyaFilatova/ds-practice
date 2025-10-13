from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

BASE_DIR = Path(__file__).resolve().parent

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / "../models/ner")
model = AutoModelForTokenClassification.from_pretrained(BASE_DIR / "../models/ner")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


ANIMAL_LIST = [
    "cat", "dog", "cow", "horse", "sheep", "elephant",
    "butterfly", "chicken", "spider", "squirrel"
]

def extract_animals(text: str):
    """Return a list of detected animals with token positions"""
    entities = ner_pipeline(text)
    results = []
    tokens = text.split()
    for ent in entities:
        word = ent["word"].lower()
        for animal in ANIMAL_LIST:
            if animal in word:
                try:
                    idx = tokens.index(animal)
                except ValueError:
                    idx = 0
                results.append({"word": animal, "index": idx})
    return tokens, results

if __name__ == "__main__":
    text = "This is not a cow"
    tokens, animals = extract_animals(text)
    print(tokens, animals)

    text = "There is a cow in the picture."
    tokens, animals = extract_animals(text)
    print(tokens, animals)
