import json
import random
from pathlib import Path
import re

# ========== CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "../data/ner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

animals = ["cat", "dog", "horse", "cow", "spider", "elephant", "butterfly", "chicken", "sheep", "squirrel"]

# Positive templates (structured)
templates_positive = [
    "There is a {} in the picture.",
    "I see a {}.",
    "A {} is running fast.",
    "This is a {}.",
    "Can you find the {}?",
    "Is that a {}?",
    "I believe this is a {}.",
    "Looks like a {} to me.",
    "Could it be a {}?",
    "The photo shows a {}."
]

# Off-template / varied phrasing
templates_varied = [
    "a {}.",
    "the {}.",
    "{}.",
    "Look, a {} over there!",
    "I think I see a {}",
    "Could that be a {} in the field?",
    "Maybe this is a {}?",
    "Is it possible that it's a {}?",
    "I spotted a {} somewhere.",
    "Seems like a {} is around.",
    "Check out that {}!",
    "Do you see the {} there?",
    "That appears to be a {}."
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
    "Not a {} in sight.",
    "Without a {}.",
    "No {} can be found."
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

def generate_sample(idx, templates):
    animal = random.choice(animals)
    template = templates[idx % len(templates)]
    remove_punctuation = random.choice([0, 1])
    sentence = template.format(animal)
    sentence = re.sub(r'[^\w\s]', '', sentence) if remove_punctuation else sentence
    sentence = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    tags = ["B-ANIMAL" if re.sub(r'[^\w\s]', '', token) == animal else "O" for token in sentence]
    return {"tokens": sentence, "ner_tags": tags}

# ========== GENERATION FUNCTION ==========
def make_samples(
    num_positive=200, 
    num_varied=100, 
    num_negation=200, 
    num_negative=50
):
    data = []

    # --- Positive samples ---
    for idx in range(num_positive):
        data.append(generate_sample(idx, templates_positive))

    # --- Varied/off-template samples ---
    for idx in range(num_varied):
        data.append(generate_sample(idx, templates_varied))

    # --- Negation samples ---
    for idx in range(num_negation):
        data.append(generate_sample(idx, templates_negation))

    # --- Negative samples (no animal) ---
    for idx in range(num_negative):
        data.append(generate_sample(idx, templates_negative))

    random.shuffle(data)
    return data

# ========== MAIN ==========
def main():
    all_data = make_samples()
    split_idx = int(0.8 * len(all_data))
    train, val = all_data[:split_idx], all_data[split_idx:]

    with open(OUTPUT_DIR / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(OUTPUT_DIR / "val.json", "w") as f:
        json.dump(val, f, indent=2)

    print(f"✅ Generated {len(train)} train and {len(val)} val samples in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
