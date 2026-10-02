import re

from image_relevance.schemas import Metadata

# Explicit taxonomy aliases are a guard constraint; ranking still uses real embeddings.
ALIASES = {
    "fox": ("fox", "foxes", "vulpes vulpes", "vulpes"),
    "wolf": ("wolf", "wolves", "canis lupus"),
    "dog": ("dog", "dogs", "canis familiaris", "canis lupus familiaris", "puppy", "puppies"),
    "bear": ("bear", "bears", "ursus", "grizzly"),
    "deer": ("deer", "elk", "cervus", "buck", "fawn"),
}


def subjects(text: str) -> set[str]:
    text = text.casefold().replace("_", " ").replace("-", " ")
    # The longer domestic-dog scientific name must not also count as a wolf.
    text = text.replace("canis lupus familiaris", "domestic dog")
    found = set()
    for subject, aliases in ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                found.add(subject)
                break
    return found


def embedding_text(text: str) -> str:
    for scientific, common in (
        ("canis lupus familiaris", "domestic dog"),
        ("vulpes vulpes", "red fox"),
        ("canis familiaris", "domestic dog"),
        ("canis lupus", "gray wolf"),
        ("ursus arctos", "brown bear"),
        ("cervus elaphus", "red deer"),
    ):
        text = re.sub(r"\b" + re.escape(scientific) + r"\b", common, text, flags=re.IGNORECASE)
    return text


def cosine(left: list[float], right: list[float]) -> float:
    # Stored embeddings have already been validated and normalized.
    score = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, score))


def guard(
    post_text: str, metadata: Metadata, similarity: float, confidence: float, threshold: float
) -> list[str]:
    post_text = post_text.casefold().replace("_", " ").replace("-", " ")
    expected = subjects(post_text)
    detected = subjects(metadata.subject)
    reasons = []

    if len(expected) != 1:
        reasons.append("Article has no single supported animal subject; manual selection required")

    elif detected != expected or metadata.category != "animal":
        reasons.append(
            f"Animal category mismatch: expected {', '.join(sorted(expected))}, "
            f"detected {metadata.subject}"
        )
    elif (
        expected == {"fox"}
        and re.search(r"\bred fox(?:es)?\b|\bvulpes vulpes\b", post_text.casefold())
        and metadata.subject.casefold().replace("_", " ").replace("-", " ") != "red fox"
    ):
        reasons.append(f"Animal species mismatch: expected red fox, detected {metadata.subject}")

    if metadata.confidence < confidence:
        reasons.append(f"Low vision confidence {metadata.confidence:.3f} < {confidence:.3f}")

    if similarity < threshold:
        reasons.append(f"Similarity below threshold: {similarity:.3f} < {threshold:.3f}")

    return reasons
