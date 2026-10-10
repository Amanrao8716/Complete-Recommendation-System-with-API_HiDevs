"""Shared fixtures: in-memory database seeded with the sample dataset."""

import pytest

from api.app import create_app
from data.database import init_db, make_engine, make_session_factory
from engine.orchestrator import RecommendationOrchestrator
from scripts.seed_data import seed_database
from tests.test_data import SAMPLE_DATASET


@pytest.fixture()
def session_factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    factory = make_session_factory(engine)
    seed_database(factory, SAMPLE_DATASET)
    return factory


@pytest.fixture()
def empty_session_factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture()
def orchestrator(session_factory):
    return RecommendationOrchestrator(session_factory, cache_ttl=60)


@pytest.fixture()
def client(orchestrator):
    app = create_app(orchestrator=orchestrator, api_key="")
    app.testing = True
    return app.test_client()
