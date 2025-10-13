from ner.infer import extract_animals
from vision.infer import classify_animal

NEGATION_WORDS = {"not", "no", "don't", "n't", "without", "never", "none", "nothing"}

def is_negated(tokens, entity_idx, window=5):
    start = max(0, entity_idx - window)
    context = tokens[start:entity_idx]
    return any(word.lower() in NEGATION_WORDS for word in context)

def verify_text_image_claim(text: str, image_path: str):
    print()
    # Step 1: NER
    tokens, detected_animals = extract_animals(text)
    
    if not detected_animals:
        print(f"Text: '{text}'")
        print("No animal detected in text.")
        print(f"Pipeline result: False")
        return False

    # Step 2: take first detected animal
    print(detected_animals)
    animal_entity = detected_animals[0]
    animal_name = animal_entity["word"]
    
    # Step 3: check negation
    negated = is_negated(tokens, animal_entity["index"])
    expected_presence = not negated

    # Step 4: Image classification
    predicted_animal = classify_animal(image_path)
    image_matches = predicted_animal.lower() == animal_name.lower()

    # Step 5: Final decision
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
    verify_text_image_claim("not a dog", "test_images/cow/0001.jpeg")
    verify_text_image_claim("I don't think it's a sheep", "test_images/dog/0001.jpeg")
    verify_text_image_claim("Look, a cat over there!", "test_images/cat/0001.jpeg")
