"""Pipeline to verify claims about animals in text and images."""
import json
import logging
from ner.infer import extract_animals
from vision.infer import classify_animal

logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_as_json(message, **kwargs):
    """Log messages in JSON format."""
    logging.info(json.dumps({"message": message, **kwargs}))

NEGATION_WORDS = {"not", "no", "don't", "n't", "without", "never", "none", "nothing"}

def process_text(text):
    """Extract tokens and detected animals from text."""
    tokens, detected_animals = extract_animals(text)
    if not detected_animals:
        log_as_json("No animals detected", text=text)
        return tokens, None
    log_as_json("Detected animals", detected_animals=detected_animals)
    return tokens, detected_animals

def handle_negation(tokens, animal_entity, window=5):
    """Check if the detected animal is negated in the text."""
    entity_idx = animal_entity["index"]
    start = max(0, entity_idx - window)
    context = tokens[start:entity_idx]
    return any(word.lower() in NEGATION_WORDS for word in context)

def verify_text_image_claim(text: str, image_path: str):
    """
    Verify the claim in the text about the presence of an animal in the image.

    Args:
        text (str): Input text describing the claim.
        image_path (str): Path to the image file.

    Returns:
        bool: True if the claim is verified, False otherwise.
    """
    log_as_json("Verifying claim", text=text, image_path=image_path)
    tokens, detected_animals = process_text(text)

    if not detected_animals:
        return False

    # Take the first detected animal
    animal_entity = detected_animals[0]
    animal_name = animal_entity["word"]

    # Check negation
    negated = handle_negation(tokens, animal_entity)
    expected_presence = not negated

    # Image classification
    predicted_animal = classify_animal(image_path)
    image_matches = predicted_animal.lower() == animal_name.lower()

    # Final decision
    result = image_matches == expected_presence

    # Debug info
    print(f"Text: '{text}'")
    print(f"Detected animal: {animal_name} | Negated: {negated}")
    print(f"Image prediction: {predicted_animal}")
    print(f"Pipeline result: {result}")

    return result

if __name__ == "__main__":
    # Examples
    verify_text_image_claim("There is a horse in the picture.", "test_images/horse/0001.jpeg")
    print("-"*40)
    verify_text_image_claim("not a dog", "test_images/cow/0001.jpeg")
    print("-"*40)
    verify_text_image_claim("I don't think it's a sheep", "test_images/dog/0001.jpeg")
    print("-"*40)
    verify_text_image_claim("Look, a cat over there!", "test_images/cat/0001.jpeg")
    print("-"*40)
