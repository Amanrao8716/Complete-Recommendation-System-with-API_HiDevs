"""Scoring: individual scorers, weighted combination, re-ranking."""

from collections import Counter
from dataclasses import dataclass, field

from engine.similarity import weighted_overlap

STRATEGY_WEIGHTS = {
    "hybrid": {
        "skill_match": 0.30,
        "category_affinity": 0.20,
        "difficulty_fit": 0.15,
        "collaborative": 0.25,
        "popularity": 0.10,
    },
    "skill_based": {
        "skill_match": 0.60,
        "difficulty_fit": 0.25,
        "category_affinity": 0.15,
    },
    "collaborative": {"collaborative": 0.80, "popularity": 0.20},
    "popularity": {"popularity": 1.0},
    "cold_start": {
        "category_affinity": 0.40,
        "skill_match": 0.30,
        "popularity": 0.20,
        "difficulty_fit": 0.10,
    },
}


@dataclass
class ScoringContext:
    """Per-request inputs shared by all scorers."""

    user: object
    cf_scores: dict
    max_popularity: float


@dataclass
class ScoredItem:
    """A content id with its final score and per-component scores."""

    content_id: int
    score: float
    components: dict = field(default_factory=dict)


class SkillMatchScorer:
    """How much of the item's skills the user needs/cares about."""

    name = "skill_match"

    def score(self, content, ctx):
        return weighted_overlap(content.skill_ids, ctx.user.need)


class CategoryAffinityScorer:
    """Affinity for the item's category (interests + behaviour)."""

    name = "category_affinity"

    def score(self, content, ctx):
        return ctx.user.category_affinity.get(content.category, 0.0)


class DifficultyFitScorer:
    """Prefers items slightly above the user's current level."""

    name = "difficulty_fit"

    def score(self, content, ctx):
        target = min(5.0, ctx.user.level + 0.5)
        return max(0.0, 1.0 - abs(content.difficulty - target) / 3.0)


class CollaborativeScorer:
    """Opinion of users with similar interaction history."""

    name = "collaborative"

    def score(self, content, ctx):
        return min(1.0, ctx.cf_scores.get(content.id, 0.0))


class PopularityScorer:
    """Global popularity normalised to ``[0, 1]``."""

    name = "popularity"

    def score(self, content, ctx):
        if ctx.max_popularity <= 0:
            return 0.0
        return max(0.0, min(1.0, content.popularity / ctx.max_popularity))


SCORERS = {
    s.name: s
    for s in (
        SkillMatchScorer(),
        CategoryAffinityScorer(),
        DifficultyFitScorer(),
        CollaborativeScorer(),
        PopularityScorer(),
    )
}


class WeightedScorer:
    """Combine scorers with (normalised) weights."""

    def __init__(self, weights, scorers=None):
        scorers = scorers or SCORERS
        unknown = set(weights) - set(scorers)
        if unknown:
            raise ValueError(f"unknown scorers: {sorted(unknown)}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.weights = {k: v / total for k, v in weights.items()}
        self._scorers = {k: scorers[k] for k in weights}

    def score(self, content, ctx):
        """Return a ``ScoredItem`` for ``content``."""
        parts = {n: s.score(content, ctx) for n, s in self._scorers.items()}
        total = sum(self.weights[n] * v for n, v in parts.items())
        return ScoredItem(content.id, total, parts)


def rank(scored, contents):
    """Sort by score, then popularity, then id (deterministic)."""
    return sorted(
        scored,
        key=lambda s: (
            -s.score,
            -contents[s.content_id].popularity,
            s.content_id,
        ),
    )


def diversify(ranked, contents, limit, penalty=0.92):
    """Greedy re-rank that gently penalises repeated categories."""
    pool = list(ranked[: max(limit * 5, limit)])
    picked, counts = [], Counter()
    while pool and len(picked) < limit:
        best = max(
            pool,
            key=lambda s: (
                s.score * penalty ** counts[contents[s.content_id].category],
                -s.content_id,
            ),
        )
        pool.remove(best)
        picked.append(best)
        counts[contents[best.content_id].category] += 1
    return picked


def _join(names):
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _skill_phrase(content, ctx, skill_names):
    ranked = sorted(content.skill_ids, key=lambda s: -ctx.user.need.get(s, 0))
    matched = [skill_names[s] for s in ranked if ctx.user.need.get(s, 0) > 0]
    return f"builds skills you care about ({_join(matched[:3])})"


def _difficulty_phrase(content, ctx, _names):
    if ctx.user.is_cold:
        return f"is beginner-friendly (difficulty {content.difficulty})"
    return f"is pitched at your level (difficulty {content.difficulty})"


_PHRASES = {
    "skill_match": _skill_phrase,
    "difficulty_fit": _difficulty_phrase,
    "category_affinity": lambda c, ctx, n: (
        f"fits your interest in {c.category}"
    ),
    "collaborative": lambda c, ctx, n: (
        "was well received by learners with similar activity"
    ),
    "popularity": lambda c, ctx, n: "is popular among learners",
}


def explain(content, item, weights, ctx, skill_names, max_reasons=2):
    """Build a short human-readable explanation from the top components."""
    contributions = sorted(
        (
            (weights.get(name, 0.0) * value, name)
            for name, value in item.components.items()
            if value > 0
        ),
        reverse=True,
    )
    phrases = [
        _PHRASES[name](content, ctx, skill_names)
        for contribution, name in contributions
        if contribution > 0.01
    ][:max_reasons]
    if not phrases:
        return "Recommended as a good starting point."
    return "Recommended because it " + " and ".join(phrases) + "."
