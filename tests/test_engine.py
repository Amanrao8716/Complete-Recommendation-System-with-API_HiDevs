"""Tests for similarity, candidates, scorers, evaluator, orchestrator."""

import math

import pytest

from data.database import session_scope
from data.repositories import UserSkillRepository
from engine import evaluator
from engine.candidate_gen import STRATEGY_SOURCES
from engine.domain import UserContext, interaction_strength
from engine.orchestrator import (
    STRATEGIES,
    ContentNotFoundError,
    InvalidFeedbackError,
    InvalidRequestError,
    RecommendationOrchestrator,
    UserNotFoundError,
)
from engine.scorer import (
    SCORERS,
    ScoredItem,
    ScoringContext,
    WeightedScorer,
    diversify,
    rank,
)
from engine.similarity import (
    cosine_similarity,
    jaccard_similarity,
    top_k_similar,
    weighted_overlap,
)

# ---------------------------------------------------------------- similarity


def test_cosine_similarity():
    assert cosine_similarity({"a": 1, "b": 1}, {"a": 1, "b": 1}) == (
        pytest.approx(1.0)
    )
    assert cosine_similarity({"a": 1}, {"b": 1}) == 0.0
    assert cosine_similarity({}, {"a": 1}) == 0.0
    assert cosine_similarity({"a": 1, "b": 0}, {"a": 1, "b": 1}) == (
        pytest.approx(1 / math.sqrt(2))
    )


def test_jaccard_and_overlap():
    assert jaccard_similarity([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert jaccard_similarity([], []) == 0.0
    assert weighted_overlap([1, 2], {1: 1.0}) == 0.5
    assert weighted_overlap([], {1: 1.0}) == 0.0


def test_top_k_similar_orders_and_excludes():
    target = {"a": 1.0}
    others = {1: {"a": 1.0}, 2: {"a": 1.0, "b": 1.0}, 3: {"b": 1.0}}
    result = top_k_similar(target, others, k=5, exclude={1})
    assert [k for k, _ in result] == [2]


def test_interaction_strength():
    assert interaction_strength("complete") == 1.0
    assert interaction_strength("dislike") == -1.0
    assert interaction_strength("rate", 5) == 1.0
    assert interaction_strength("rate", 1) == -1.0
    assert interaction_strength("rate") == 0.0
    assert interaction_strength("like", 5) == pytest.approx(0.85)


# ------------------------------------------------------------------- scorers


def _ctx(**kw):
    base = dict(
        user_id=1,
        name="t",
        interests=[],
        proficiency={},
        strengths={},
        need={},
        category_affinity={},
        level=1.0,
    )
    base.update(kw)
    return UserContext(**base)


def test_individual_scorers(orchestrator):
    item = orchestrator._contents[4]  # React Fundamentals, difficulty 3
    sc = ScoringContext(
        _ctx(
            need={item.id: 0.0},
            level=2.5,
            category_affinity={item.category: 0.7},
        ),
        {item.id: 0.4},
        100.0,
    )
    assert SCORERS["difficulty_fit"].score(item, sc) == pytest.approx(1.0)
    assert SCORERS["category_affinity"].score(item, sc) == 0.7
    assert SCORERS["collaborative"].score(item, sc) == 0.4
    assert SCORERS["popularity"].score(item, sc) == pytest.approx(0.88)
    assert SCORERS["skill_match"].score(item, sc) == 0.0
    sc.max_popularity = 0
    assert SCORERS["popularity"].score(item, sc) == 0.0


def test_weighted_scorer_validation_and_normalisation():
    with pytest.raises(ValueError):
        WeightedScorer({"nope": 1.0})
    with pytest.raises(ValueError):
        WeightedScorer({"popularity": 0.0})
    scorer = WeightedScorer({"popularity": 2.0, "collaborative": 2.0})
    assert sum(scorer.weights.values()) == pytest.approx(1.0)


def test_rank_and_diversify_prefer_variety(orchestrator):
    contents = orchestrator._contents
    web = [ScoredItem(i, 0.80) for i in (1, 2, 3)]
    data = [ScoredItem(7, 0.78)]
    ranked = rank(web + data, contents)
    assert [s.content_id for s in ranked] == [3, 1, 2, 7]  # tie -> popularity
    picked = diversify(ranked, contents, limit=2, penalty=0.9)
    assert {contents[s.content_id].category for s in picked} == {
        "Web Development",
        "Data Science",
    }


# ----------------------------------------------------------------- evaluator


def test_ranking_metrics_hand_computed():
    rec, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert evaluator.precision_at_k(rec, rel, 5) == pytest.approx(0.4)
    assert evaluator.recall_at_k(rec, rel, 5) == pytest.approx(2 / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert evaluator.ndcg_at_k(rec, rel, 5) == pytest.approx(dcg / idcg)
    assert evaluator.ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)
    assert evaluator.precision_at_k(rec, rel, 0) == 0.0
    assert evaluator.recall_at_k(rec, set(), 5) == 0.0
    assert evaluator.ndcg_at_k(rec, set(), 5) == 0.0


def test_evaluate_rankings_and_coverage():
    result = evaluator.evaluate_rankings(
        {1: [1, 2], 2: [3, 4]}, {1: {1}, 2: {9}, 3: set()}, k=2
    )
    assert result["users"] == 2
    assert result["precision"] == pytest.approx(0.25)
    assert evaluator.evaluate_rankings({}, {}, 5)["users"] == 0
    assert evaluator.catalog_coverage({1: [1, 2], 2: [2, 3]}, 6) == 0.5
    assert evaluator.catalog_coverage({}, 0) == 0.0


# ------------------------------------------------------------- orchestrator


def test_recommendation_shape_and_limit(orchestrator):
    recs = orchestrator.get_recommendations(1, limit=4)
    assert len(recs) == 4
    first = recs[0]
    assert {
        "content_id",
        "title",
        "score",
        "explanation",
        "components",
    } <= set(first)
    assert first["explanation"].startswith("Recommended because")
    scores = [r["score"] for r in recs]
    assert all(0 <= s <= 1 for s in scores)


def test_seen_items_are_never_recommended(orchestrator):
    seen = set(orchestrator._strengths[1])
    for strategy in STRATEGIES:
        ids = {
            r["content_id"]
            for r in orchestrator.get_recommendations(1, 10, strategy)
        }
        assert not ids & seen, strategy


def test_recommendations_are_relevant_to_interests(orchestrator):
    recs = orchestrator.get_recommendations(1, 3)  # web learner
    assert [r["category"] for r in recs][:2] == ["Web Development"] * 2
    recs = orchestrator.get_recommendations(8, 3)  # ML learner
    assert recs[0]["category"] == "Machine Learning"
    recs = orchestrator.get_recommendations(10, 3)  # cloud learner
    assert recs[0]["category"] == "Cloud & DevOps"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_works(orchestrator, strategy):
    result = orchestrator.get_recommendations_detailed(5, 5, strategy)
    assert result.strategy == strategy
    assert len(result.items) == 5


def test_cold_start_with_interests(orchestrator):
    result = orchestrator.get_recommendations_detailed(13, 3)
    assert result.strategy == "cold_start"
    assert all(i.category == "Web Development" for i in result.items)


def test_cold_start_blank_user_gets_popular_beginner_content(orchestrator):
    result = orchestrator.get_recommendations_detailed(14, 3)
    assert result.strategy == "cold_start"
    assert result.items[0].title == "Python for Beginners"
    assert all(i.difficulty <= 2 for i in result.items)
    assert "beginner-friendly" in result.items[0].explanation


def test_limit_larger_than_unseen_catalog(orchestrator):
    result = orchestrator.get_recommendations_detailed(1, 50)
    assert len(result.items) == 24 - len(orchestrator._strengths[1])


def test_validation_errors(orchestrator):
    with pytest.raises(UserNotFoundError):
        orchestrator.get_recommendations(999)
    with pytest.raises(InvalidRequestError):
        orchestrator.get_recommendations(1, limit=0)
    with pytest.raises(InvalidRequestError):
        orchestrator.get_recommendations(1, limit="5")
    with pytest.raises(InvalidRequestError):
        orchestrator.get_recommendations(1, strategy="magic")


def test_cache_hit_and_expiry(session_factory):
    now = [0.0]
    orch = RecommendationOrchestrator(
        session_factory, cache_ttl=10, clock=lambda: now[0]
    )
    first = orch.get_recommendations_detailed(1, 5)
    second = orch.get_recommendations_detailed(1, 5)
    assert (first.cached, second.cached) == (False, True)
    now[0] = 11.0
    assert orch.get_recommendations_detailed(1, 5).cached is False
    stats = orch.stats()["cache"]
    assert stats["hits"] == 1 and stats["misses"] == 2
    assert (
        orch.get_recommendations_detailed(1, 5, use_cache=False).cached
        is False
    )


def test_cache_can_be_disabled_and_is_bounded(session_factory):
    orch = RecommendationOrchestrator(session_factory, cache_ttl=0)
    orch.get_recommendations(1)
    assert orch.stats()["cache"]["size"] == 0
    small = RecommendationOrchestrator(session_factory, cache_max_size=2)
    for uid in (1, 2, 3):
        small.get_recommendations(uid)
    assert small.stats()["cache"]["size"] <= 2


def test_feedback_invalidates_cache_and_excludes_item(orchestrator):
    top = orchestrator.get_recommendations(1, 3)[0]["content_id"]
    assert orchestrator.get_recommendations_detailed(1, 3).cached
    record = orchestrator.record_feedback(1, top, "complete")
    assert record["strength"] == 1.0 and record["interaction_id"] > 0
    fresh = orchestrator.get_recommendations_detailed(1, 3)
    assert not fresh.cached
    assert top not in [i.content_id for i in fresh.items]


def test_feedback_moves_user_out_of_cold_start(orchestrator):
    assert orchestrator.get_recommendations_detailed(14, 3).strategy == (
        "cold_start"
    )
    orchestrator.record_feedback(14, 19, "complete")  # Linux Command Line
    after = orchestrator.get_recommendations_detailed(14, 5)
    assert after.strategy == "hybrid"
    assert after.items[0].category == "Cloud & DevOps"


def test_feedback_changes_ranking_towards_liked_topic(orchestrator):
    before = [r["category"] for r in orchestrator.get_recommendations(14, 5)]
    for content_id in (21, 22, 23):  # three cloud items
        orchestrator.record_feedback(14, content_id, "like")
    after = [r["category"] for r in orchestrator.get_recommendations(14, 5)]
    assert after.count("Cloud & DevOps") > before.count("Cloud & DevOps")


def test_negative_feedback_lowers_category(orchestrator):
    base = orchestrator._build_context(13).category_affinity["Web Development"]
    orchestrator.record_feedback(13, 1, "dislike")
    orchestrator.record_feedback(13, 2, "rate", 1)
    lowered = orchestrator._build_context(13).category_affinity.get(
        "Web Development", 0.0
    )
    assert lowered < base


def test_complete_raises_proficiency_and_persists(
    orchestrator, session_factory
):
    ids = orchestrator._contents[19].skill_ids  # Linux Command Line
    skill_id = next(iter(ids))
    orchestrator.record_feedback(14, 19, "complete")
    assert orchestrator._user_skills[14][skill_id] == pytest.approx(0.15)
    orchestrator.record_feedback(14, 20, "complete")
    with session_scope(session_factory) as session:
        stored = UserSkillRepository(session).get_for_user(14)
    assert stored[skill_id] == pytest.approx(0.2775)


def test_view_does_not_overwrite_stronger_signal(orchestrator):
    orchestrator.record_feedback(1, 1, "view")
    assert orchestrator._strengths[1][1] == 1.0  # still the "complete"


def test_feedback_validation(orchestrator):
    with pytest.raises(InvalidFeedbackError):
        orchestrator.record_feedback(1, 5, "share")
    with pytest.raises(InvalidFeedbackError):
        orchestrator.record_feedback(1, 5, "rate")
    with pytest.raises(InvalidFeedbackError):
        orchestrator.record_feedback(1, 5, "like", rating=9)
    with pytest.raises(InvalidFeedbackError):
        orchestrator.record_feedback(1, 5, "like", rating=True)
    with pytest.raises(InvalidFeedbackError):
        orchestrator.record_feedback(1, 5, "like", rating="5")
    with pytest.raises(UserNotFoundError):
        orchestrator.record_feedback(999, 5, "like")
    with pytest.raises(ContentNotFoundError):
        orchestrator.record_feedback(1, 999, "like")


def test_user_created_after_startup_is_picked_up(
    orchestrator, session_factory
):
    from data.repositories import UserRepository

    with session_scope(session_factory) as session:
        new_id = UserRepository(session).create("Late", "aws").id
    result = orchestrator.get_recommendations_detailed(new_id, 3)
    assert result.items[0].category == "Cloud & DevOps"


def test_explanations_name_real_skills(orchestrator):
    rec = orchestrator.get_recommendations(5, 1)[0]
    assert "builds skills you care about" in rec["explanation"]


def test_collaborative_uses_similar_users(orchestrator):
    ctx = orchestrator._build_context(11)
    neighbors = orchestrator._generator.find_neighbors(ctx)
    assert neighbors and neighbors[0][0] == 10  # Jai behaves like Kavya


def test_strategy_sources_cover_all_strategies():
    assert set(STRATEGY_SOURCES) == set(STRATEGIES)


def test_stats_shape(orchestrator):
    stats = orchestrator.stats()
    assert stats["users"] == 14 and stats["content_items"] == 24
    assert stats["interactions"] == 40
