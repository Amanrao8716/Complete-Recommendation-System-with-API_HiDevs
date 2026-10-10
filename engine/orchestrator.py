"""RecommendationOrchestrator: wires data, engine components and cache."""

import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import replace

from data.database import session_scope
from data.repositories import (
    ContentRepository,
    ContentSkillRepository,
    InteractionRepository,
    SkillRepository,
    UserRepository,
    UserSkillRepository,
)
from engine.candidate_gen import CandidateGenerator
from engine.domain import (
    INTERACTION_TYPES,
    POSITIVE_THRESHOLD,
    ContentInfo,
    Recommendation,
    RecommendationResult,
    UserContext,
    UserInfo,
    interaction_strength,
)
from engine.scorer import (
    STRATEGY_WEIGHTS,
    ScoringContext,
    WeightedScorer,
    diversify,
    explain,
    rank,
)

logger = logging.getLogger(__name__)

MAX_LIMIT = 50
STRATEGIES = tuple(STRATEGY_WEIGHTS)
VALID_STRATEGIES = ("auto",) + STRATEGIES
PROFICIENCY_GAIN = 0.15


class UserNotFoundError(LookupError):
    """Raised when a user id does not exist."""


class ContentNotFoundError(LookupError):
    """Raised when a content id does not exist."""


class InvalidFeedbackError(ValueError):
    """Raised when feedback fails validation."""


class InvalidRequestError(ValueError):
    """Raised when limit/strategy arguments are invalid."""


def _merge_strength(strengths, user_id, content_id, type_, rating):
    """Store the interaction signal; views never overwrite stronger ones."""
    per_user = strengths.setdefault(user_id, {})
    if type_ == "view" and content_id in per_user:
        return
    per_user[content_id] = interaction_strength(type_, rating)


def _clamp(value):
    return max(0.0, min(1.0, value))


class RecommendationOrchestrator:
    """Serve recommendations from an in-memory snapshot of the database.

    Strategies: ``hybrid``, ``skill_based``, ``collaborative``,
    ``popularity`` and ``cold_start`` (``auto`` picks between hybrid
    and cold start).  Results are cached per (user, limit, strategy)
    and invalidated whenever the user submits feedback.
    """

    def __init__(
        self,
        session_factory,
        strategy_weights=None,
        cache_ttl=300.0,
        cache_max_size=5000,
        diversity_penalty=0.92,
        clock=time.monotonic,
    ):
        self._session_factory = session_factory
        self._weights = {
            name: dict(w)
            for name, w in (strategy_weights or STRATEGY_WEIGHTS).items()
        }
        self.cache_ttl = cache_ttl
        self.cache_max_size = cache_max_size
        self.diversity_penalty = diversity_penalty
        self._clock = clock
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_index = defaultdict(set)
        self._hits = 0
        self._misses = 0
        self._n_interactions = 0
        self.refresh()

    @property
    def session_factory(self):
        """The SQLAlchemy session factory in use."""
        return self._session_factory

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def refresh(self):
        """(Re)load every table into the in-memory snapshot."""
        with session_scope(self._session_factory) as session:
            skills = {s.id: s.name for s in SkillRepository(session).get_all()}
            links = defaultdict(set)
            for content_id, skill_id in ContentSkillRepository(
                session
            ).get_all():
                links[content_id].add(skill_id)
            contents = {
                c.id: ContentInfo(
                    id=c.id,
                    title=c.title,
                    category=c.category,
                    difficulty=c.difficulty,
                    popularity=float(c.popularity or 0.0),
                    skill_ids=frozenset(links[c.id]),
                )
                for c in ContentRepository(session).get_all()
            }
            users = {
                u.id: UserInfo(u.id, u.name, tuple(u.interest_list()))
                for u in UserRepository(session).get_all()
            }
            user_skills = defaultdict(dict)
            for row in UserSkillRepository(session).get_all():
                user_skills[row.user_id][row.skill_id] = row.proficiency
            interactions = InteractionRepository(session).get_all()
            raw = [
                (i.user_id, i.content_id, i.type, i.rating)
                for i in interactions
            ]

        strengths = {}
        for user_id, content_id, type_, rating in raw:
            _merge_strength(strengths, user_id, content_id, type_, rating)

        with self._lock:
            self._skills = skills
            self._skill_words = {
                sid: (name.lower(), set(name.lower().split()))
                for sid, name in skills.items()
            }
            self._contents = contents
            self._users = users
            self._user_skills = user_skills
            self._strengths = strengths
            self._n_interactions = len(raw)
            self._max_popularity = max(
                (c.popularity for c in contents.values()), default=0.0
            )
            self._generator = CandidateGenerator(contents, strengths)
            self._cache.clear()
            self._cache_index.clear()

    def _ensure_user(self, user_id):
        if user_id in self._users:
            return
        with session_scope(self._session_factory) as session:
            exists = UserRepository(session).exists(user_id)
        if not exists:
            raise UserNotFoundError(f"user {user_id} not found")
        self.refresh()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def _interest_match(self, tokens, name_lower, words):
        return any(t == name_lower or t in words for t in tokens)

    def _build_context(self, user_id):
        user = self._users[user_id]
        strengths = dict(self._strengths.get(user_id, {}))
        proficiency = dict(self._user_skills.get(user_id, {}))
        tokens = list(user.interests)

        pos_skill, neg_skill = Counter(), Counter()
        pos_cat, neg_cat = Counter(), Counter()
        for content_id, strength in strengths.items():
            item = self._contents.get(content_id)
            if item is None or strength == 0:
                continue
            skill_bucket = pos_skill if strength > 0 else neg_skill
            cat_bucket = pos_cat if strength > 0 else neg_cat
            for skill_id in item.skill_ids:
                skill_bucket[skill_id] += abs(strength)
            cat_bucket[item.category] += abs(strength)

        need = {}
        for skill_id, (name_lower, words) in self._skill_words.items():
            interest = (
                1.0 if self._interest_match(tokens, name_lower, words) else 0.0
            )
            prof = proficiency.get(skill_id, 0.0)
            affinity = _clamp(
                0.6 * interest
                + 0.6 * min(1.0, pos_skill[skill_id] / 2)
                - 0.4 * min(1.0, neg_skill[skill_id] / 2)
                + 0.25 * prof
            )
            if affinity > 0:
                need[skill_id] = affinity * (1.0 - 0.5 * prof)

        category_affinity = {}
        for category in {c.category for c in self._contents.values()}:
            words = set(category.lower().split())
            interest = (
                1.0
                if self._interest_match(tokens, category.lower(), words)
                else 0.0
            )
            affinity = _clamp(
                0.6 * interest
                + 0.6 * min(1.0, pos_cat[category] / 2)
                - 0.4 * min(1.0, neg_cat[category] / 2)
            )
            if affinity > 0:
                category_affinity[category] = affinity

        levels = []
        if proficiency:
            levels.append(
                1.0 + 4.0 * sum(proficiency.values()) / len(proficiency)
            )
        liked = [
            self._contents[c].difficulty
            for c, s in strengths.items()
            if s >= POSITIVE_THRESHOLD and c in self._contents
        ]
        if liked:
            levels.append(sum(liked) / len(liked))
        level = sum(levels) / len(levels) if levels else 1.0

        return UserContext(
            user_id=user_id,
            name=user.name,
            interests=tokens,
            proficiency=proficiency,
            strengths=strengths,
            need=need,
            category_affinity=category_affinity,
            level=level,
        )

    def _weights_for(self, strategy, ctx):
        weights = dict(self._weights[strategy])
        if strategy == "hybrid" and len(ctx.strengths) < 3:
            weights["collaborative"] = weights.get("collaborative", 0) * 0.5
        return weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_recommendations(self, user_id, limit=5, strategy="auto"):
        """Return a list of recommendation dictionaries."""
        result = self.get_recommendations_detailed(user_id, limit, strategy)
        return [item.to_dict() for item in result.items]

    def get_recommendations_detailed(
        self, user_id, limit=5, strategy="auto", use_cache=True
    ):
        """Return a ``RecommendationResult`` (items, strategy, cached)."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidRequestError("limit must be an integer")
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidRequestError(
                f"limit must be between 1 and {MAX_LIMIT}"
            )
        if strategy not in VALID_STRATEGIES:
            raise InvalidRequestError(
                f"strategy must be one of {', '.join(VALID_STRATEGIES)}"
            )

        key = (user_id, limit, strategy)
        with self._lock:
            if use_cache:
                entry = self._cache.get(key)
                if entry and entry[0] > self._clock():
                    self._hits += 1
                    return replace(entry[1], cached=True)
            self._misses += 1

            self._ensure_user(user_id)
            ctx = self._build_context(user_id)
            resolved = strategy
            if strategy == "auto":
                resolved = "cold_start" if ctx.is_cold else "hybrid"

            candidates = self._generator.generate(
                ctx, resolved, min_candidates=limit
            )
            weights = self._weights_for(resolved, ctx)
            scorer = WeightedScorer(weights)
            scoring_ctx = ScoringContext(
                ctx, candidates.cf_scores, self._max_popularity
            )
            scored = [
                scorer.score(self._contents[cid], scoring_ctx)
                for cid in candidates.candidates
            ]
            top = diversify(
                rank(scored, self._contents),
                self._contents,
                limit,
                self.diversity_penalty,
            )
            items = []
            for item in top:
                content = self._contents[item.content_id]
                items.append(
                    Recommendation(
                        content_id=content.id,
                        title=content.title,
                        category=content.category,
                        difficulty=content.difficulty,
                        score=round(item.score, 4),
                        strategy=resolved,
                        explanation=explain(
                            content,
                            item,
                            scorer.weights,
                            scoring_ctx,
                            self._skills,
                        ),
                        components={
                            k: round(v, 4) for k, v in item.components.items()
                        },
                    )
                )
            result = RecommendationResult(user_id, resolved, items, False)
            if use_cache and self.cache_ttl > 0:
                self._store(key, result)
            return result

    def _store(self, key, result):
        if len(self._cache) >= self.cache_max_size:
            self._cache.clear()
            self._cache_index.clear()
        self._cache[key] = (self._clock() + self.cache_ttl, result)
        self._cache_index[key[0]].add(key)

    def _invalidate_user(self, user_id):
        for key in self._cache_index.pop(user_id, set()):
            self._cache.pop(key, None)

    def record_feedback(self, user_id, content_id, type_, rating=None):
        """Persist feedback and update the in-memory model immediately.

        Completing a content item also raises the user's proficiency in
        the skills it teaches.
        """
        if type_ not in INTERACTION_TYPES:
            raise InvalidFeedbackError(
                f"type must be one of {', '.join(INTERACTION_TYPES)}"
            )
        if rating is not None:
            valid = isinstance(rating, (int, float)) and not isinstance(
                rating, bool
            )
            if not valid or not 1 <= rating <= 5:
                raise InvalidFeedbackError(
                    "rating must be a number from 1 to 5"
                )
        if type_ == "rate" and rating is None:
            raise InvalidFeedbackError(
                "rating is required when type is 'rate'"
            )

        with self._lock:
            self._ensure_user(user_id)
            content = self._contents.get(content_id)
            if content is None:
                raise ContentNotFoundError(f"content {content_id} not found")

            updates = {}
            current = self._user_skills.get(user_id, {})
            with session_scope(self._session_factory) as session:
                row = InteractionRepository(session).record(
                    user_id, content_id, type_, rating
                )
                if type_ == "complete":
                    skills_repo = UserSkillRepository(session)
                    for skill_id in content.skill_ids:
                        old = current.get(skill_id, 0.0)
                        new = round(old + PROFICIENCY_GAIN * (1.0 - old), 4)
                        skills_repo.set_proficiency(user_id, skill_id, new)
                        updates[skill_id] = new
                record = {
                    "interaction_id": row.id,
                    "user_id": user_id,
                    "content_id": content_id,
                    "type": type_,
                    "rating": rating,
                    "strength": interaction_strength(type_, rating),
                    "created_at": row.created_at.isoformat(),
                }

            _merge_strength(
                self._strengths, user_id, content_id, type_, rating
            )
            if updates:
                self._user_skills.setdefault(user_id, {}).update(updates)
            self._n_interactions += 1
            self._invalidate_user(user_id)
            return record

    def stats(self):
        """Return cache and dataset statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "cache": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": round(self._hits / total, 4) if total else 0.0,
                    "size": len(self._cache),
                    "ttl_seconds": self.cache_ttl,
                },
                "users": len(self._users),
                "content_items": len(self._contents),
                "interactions": self._n_interactions,
            }
