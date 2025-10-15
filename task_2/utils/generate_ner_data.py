"""Generate synthetic NER data with various templates and structures (improved)."""
import json
import logging
import random
from pathlib import Path
import re

# Structured JSON logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_as_json(message, **kwargs):
    logging.info(json.dumps({"message": message, **kwargs}))

# ========== CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "../data/ner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

animals = [
    "cat", "dog", "horse", "cow", "spider", "elephant",
    "butterfly", "chicken", "sheep", "squirrel"
]

# --- Templates ---
templates_positive = [
    "There is a {} in the picture.",
    "I see a {}.",
    "A {} is running fast.",
    "This is a {}.",
    "Can you find the {}?",
    "Is that a {}?",
    "Looks like a {} to me.",
    "Could it be a {}?",
    "The photo shows a {}."
]

# Very short templates (1-3 words) for improving short sentence recognition
templates_short = [
    "{}", "a {}", "the {}", "Look, a {}!", "Not a {}.", "No {} here."
]

# Negation templates
templates_negation = [
    "not a {}.",
    "not the {}.",
    "This is not a {}.",
    "I don't see a {}.",
    "There is no {} here.",
    "It isn't a {}.",
    "Never saw a {}.",
    "Not a {} in sight."
]

# Negative templates (no animal)
templates_negative = [
    "There is nothing in the photo.",
    "This picture only shows trees.",
    "Looks like an empty field.",
    "The image is just sky and clouds.",
    "Only a lake is visible.",
    "The photo shows buildings, not animals."
]

# -----------------------------
# Helper functions
# -----------------------------
def random_capitalize(word):
    """Randomly capitalize the word for augmentation."""
    return word.upper() if random.random() < 0.2 else word

def tokenize_sentence(sentence):
    """Split sentence into tokens, preserving punctuation."""
    return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)

def generate_sample(template_list, label_animal=True):
    """Generate a single NER sample from a template list."""
    template = random.choice(template_list)
    animal = random.choice(animals)
    sentence = template.format(animal)

    # Randomly remove or keep punctuation
    if random.random() < 0.3:
        sentence = re.sub(r'[^\w\s]', '', sentence)

    # Tokenize
    tokens = tokenize_sentence(sentence)
    tokens = [random_capitalize(t) for t in tokens]

    # Assign labels
    if label_animal:
        tags = ["B-ANIMAL" if t.lower() == animal else "O" for t in tokens]
    else:
        tags = ["O"] * len(tokens)

    return {"tokens": tokens, "ner_tags": tags}

def make_samples(
    num_positive=200,
    num_short=100,
    num_negation=200,
    num_negative=50
):
    """Generate samples for NER dataset."""
    data = []

    # Positive sentences
    for _ in range(num_positive):
        data.append(generate_sample(templates_positive))

    # Short templates
    for _ in range(num_short):
        data.append(generate_sample(templates_short))

    # Negation sentences
    for _ in range(num_negation):
        data.append(generate_sample(templates_negation))

    # Negative sentences (no animal)
    for _ in range(num_negative):
        data.append(generate_sample(templates_negative, label_animal=False))

    random.shuffle(data)
    return data

# -----------------------------
# MAIN
# -----------------------------
def main():
    log_as_json("Starting NER data generation")
    all_data = make_samples()

    # Split into train/val
    split_idx = int(0.8 * len(all_data))
    train, val = all_data[:split_idx], all_data[split_idx:]

    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(OUTPUT_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)

    print(f"✅ Generated {len(train)} train and {len(val)} val samples in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
