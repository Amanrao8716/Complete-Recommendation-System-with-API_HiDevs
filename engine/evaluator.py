"""Offline ranking metrics: precision@k, recall@k, NDCG@k."""

from math import log2


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k that is relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items found in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Binary-relevance normalised discounted cumulative gain."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1.0 / log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def evaluate_rankings(recommended, relevant, k=5):
    """Average the metrics over users.

    ``recommended`` maps user -> ranked list, ``relevant`` maps user ->
    set of held-out relevant items.
    """
    users = [u for u in relevant if relevant[u]]
    if not users:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "users": 0}
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    for user in users:
        ranked = recommended.get(user, [])
        totals["precision"] += precision_at_k(ranked, relevant[user], k)
        totals["recall"] += recall_at_k(ranked, relevant[user], k)
        totals["ndcg"] += ndcg_at_k(ranked, relevant[user], k)
    result = {name: value / len(users) for name, value in totals.items()}
    result["users"] = len(users)
    return result


def catalog_coverage(recommended, catalog_size):
    """Share of the catalog that appears in at least one list."""
    if catalog_size <= 0:
        return 0.0
    seen = {item for ranked in recommended.values() for item in ranked}
    return len(seen) / catalog_size
