"""Tests for the data layer (models, database helpers, repositories)."""

import pytest
from sqlalchemy.exc import IntegrityError

from data.database import make_engine, session_scope
from data.repositories import (
    ContentRepository,
    ContentSkillRepository,
    InteractionRepository,
    SkillRepository,
    UserRepository,
    UserSkillRepository,
)


def test_all_six_tables_exist(session_factory):
    from sqlalchemy import inspect

    with session_scope(session_factory) as session:
        names = set(inspect(session.get_bind()).get_table_names())
    assert {
        "users",
        "content",
        "skills",
        "user_skills",
        "content_skills",
        "interactions",
    } <= names


def test_user_crud_and_interest_list(empty_session_factory):
    with session_scope(empty_session_factory) as session:
        repo = UserRepository(session)
        user = repo.create("Zed", " Python , SQL,, ")
        assert repo.get(user.id).name == "Zed"
        assert user.interest_list() == ["python", "sql"]
        assert repo.exists(user.id) and not repo.exists(999)
        assert repo.count() == 1
        assert [u.id for u in repo.get_all()] == [user.id]


def test_skill_get_or_create_is_idempotent(empty_session_factory):
    with session_scope(empty_session_factory) as session:
        repo = SkillRepository(session)
        first = repo.get_or_create("Python")
        assert repo.get_or_create("Python").id == first.id
        assert repo.get(first.id).name == "Python"
        assert repo.get_by_name("Nope") is None
        assert len(repo.get_all()) == 1


def test_content_by_skills(session_factory):
    with session_scope(session_factory) as session:
        skills = SkillRepository(session)
        docker = skills.get_by_name("Docker").id
        items = ContentRepository(session).get_by_skills([docker])
        titles = {c.title for c in items}
        assert "Docker Essentials" in titles
        assert "CI/CD Pipelines" in titles
        assert ContentRepository(session).get_by_skills([]) == []


def test_content_skill_links(session_factory):
    with session_scope(session_factory) as session:
        repo = ContentSkillRepository(session)
        before = len(repo.get_all())
        repo.link(1, repo.get_skill_ids(1)[0])  # duplicate link: no-op
        assert len(repo.get_all()) == before
        assert len(repo.get_skill_ids(1)) == 2


def test_user_skill_upsert_and_clamp(session_factory):
    with session_scope(session_factory) as session:
        repo = UserSkillRepository(session)
        skill_id = SkillRepository(session).get_by_name("Python").id
        repo.set_proficiency(13, skill_id, 0.4)
        repo.set_proficiency(13, skill_id, 5.0)
        assert repo.get_for_user(13) == {skill_id: 1.0}
        assert len(repo.get_all()) > 1


def test_interaction_history_newest_first(session_factory):
    with session_scope(session_factory) as session:
        repo = InteractionRepository(session)
        repo.record(14, 1, "view")
        repo.record(14, 2, "like")
        history = repo.get_user_history(14)
        assert [h.content_id for h in history] == [2, 1]
        assert len(repo.get_user_history(14, limit=1)) == 1
        assert repo.count_for_user(14) == 2
        assert len(repo.get_all()) >= 42


def test_database_constraints(session_factory):
    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            ContentRepository(session).create("Bad", "X", difficulty=9)
    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            InteractionRepository(session).record(1, 1, "view", rating=9)
    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            InteractionRepository(session).record(999, 1, "view")


def test_session_scope_rolls_back_on_error(empty_session_factory):
    with pytest.raises(RuntimeError):
        with session_scope(empty_session_factory) as session:
            UserRepository(session).create("Temp")
            raise RuntimeError("boom")
    with session_scope(empty_session_factory) as session:
        assert UserRepository(session).count() == 0


def test_make_engine_uses_environment(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'x.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert str(make_engine().url) == url
