"""Similarity measures used by candidate generation and scoring."""

from math import sqrt


def cosine_similarity(a, b):
    """Cosine similarity between two sparse ``{key: value}`` vectors."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * large[k] for k, v in small.items() if k in large)
    if dot == 0:
        return 0.0
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


def jaccard_similarity(a, b):
    """Jaccard similarity between two collections treated as sets."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def weighted_overlap(items, weights):
    """Average weight of ``items`` looked up in ``weights`` (0 if absent)."""
    items = list(items)
    if not items:
        return 0.0
    return sum(weights.get(i, 0.0) for i in items) / len(items)


def top_k_similar(target, others, k, min_similarity=0.0, exclude=None):
    """Return the ``k`` most similar vectors as ``[(key, similarity)]``.

    Only similarities strictly above ``min_similarity`` are kept.
    """
    exclude = exclude or ()
    scored = []
    for key, vector in others.items():
        if key in exclude:
            continue
        sim = cosine_similarity(target, vector)
        if sim > min_similarity:
            scored.append((key, sim))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]
