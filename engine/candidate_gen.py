"""Candidate generation: decide which items are worth scoring."""

from collections import defaultdict
from dataclasses import dataclass, field

from engine.similarity import top_k_similar

STRATEGY_SOURCES = {
    "hybrid": frozenset({"skill", "category", "collaborative", "popular"}),
    "skill_based": frozenset({"skill", "category"}),
    "collaborative": frozenset({"collaborative", "popular"}),
    "popularity": frozenset({"all"}),
    "cold_start": frozenset({"skill", "category", "popular"}),
}


@dataclass
class CandidateSet:
    """Candidate ids with the sources that proposed them."""

    candidates: dict = field(default_factory=dict)
    cf_scores: dict = field(default_factory=dict)
    neighbors: list = field(default_factory=list)


class CandidateGenerator:
    """Generate candidates from skills, categories, neighbours, popularity.

    ``contents`` and ``strengths`` are shared (live) references owned by
    the orchestrator, so feedback is visible without rebuilding.
    """

    def __init__(self, contents, strengths, n_neighbors=10, n_popular=20):
        self.contents = contents
        self.strengths = strengths
        self.n_neighbors = n_neighbors
        self.n_popular = n_popular
        self._by_skill = defaultdict(list)
        self._by_category = defaultdict(list)
        for item in contents.values():
            for skill_id in item.skill_ids:
                self._by_skill[skill_id].append(item.id)
            self._by_category[item.category].append(item.id)
        ordered = sorted(
            contents.values(), key=lambda c: (-c.popularity, c.id)
        )
        self._popular = [c.id for c in ordered]

    def find_neighbors(self, ctx):
        """Return users whose interaction vectors resemble ``ctx``'s."""
        if not ctx.strengths:
            return []
        return top_k_similar(
            ctx.strengths,
            self.strengths,
            self.n_neighbors,
            exclude={ctx.user_id},
        )

    def collaborative_scores(self, ctx, neighbors):
        """Similarity-weighted opinion of neighbours, in ``[0, 1]``."""
        total = sum(sim for _, sim in neighbors)
        if total <= 0:
            return {}
        raw = defaultdict(float)
        for user_id, sim in neighbors:
            for content_id, strength in self.strengths[user_id].items():
                if content_id not in ctx.strengths:
                    raw[content_id] += sim * strength
        return {cid: v / total for cid, v in raw.items() if v > 0}

    def generate(self, ctx, strategy, min_candidates=0):
        """Return unseen candidate items for ``strategy``."""
        sources = STRATEGY_SOURCES[strategy]
        found = defaultdict(set)

        if "skill" in sources:
            for skill_id, need in ctx.need.items():
                if need > 0:
                    for cid in self._by_skill.get(skill_id, ()):
                        found[cid].add("skill")
        if "category" in sources:
            for category, aff in ctx.category_affinity.items():
                if aff > 0:
                    for cid in self._by_category.get(category, ()):
                        found[cid].add("category")

        neighbors, cf_scores = [], {}
        if "collaborative" in sources:
            neighbors = self.find_neighbors(ctx)
            cf_scores = self.collaborative_scores(ctx, neighbors)
            for cid in cf_scores:
                found[cid].add("collaborative")

        if "all" in sources:
            for cid in self._popular:
                found[cid].add("popular")
        elif "popular" in sources:
            for cid in self._popular[: self.n_popular + len(ctx.strengths)]:
                found[cid].add("popular")

        candidates = {c: s for c, s in found.items() if c not in ctx.strengths}
        if len(candidates) < min_candidates:
            for cid in self._popular:
                if cid not in ctx.strengths and cid not in candidates:
                    candidates[cid] = {"fallback"}
                if len(candidates) >= min_candidates:
                    break
        return CandidateSet(candidates, cf_scores, neighbors)
