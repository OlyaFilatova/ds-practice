NEGATION_WORDS = {"not", "no", "n't", "without", "never", "none", "nothing"}

def is_negated(tokens, entity_idx, window=5):
    """
    Detects if the entity at entity_idx is negated.
    - window: how many tokens before the entity to check
    """
    start = max(0, entity_idx - window)
    context = tokens[start:entity_idx]
    return any(word.lower() in NEGATION_WORDS for word in context)

