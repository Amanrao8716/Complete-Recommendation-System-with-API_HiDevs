"""Database connection helpers (SQLite through SQLAlchemy)."""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.models import Base

DEFAULT_DATABASE_URL = "sqlite:///recsys.db"
_IN_MEMORY_URLS = ("sqlite://", "sqlite:///:memory:")


def make_engine(url=None):
    """Create an engine.

    Falls back to the ``DATABASE_URL`` environment variable and then to a
    local ``recsys.db`` file.  In-memory SQLite databases use a static
    pool so every thread sees the same database.
    """
    url = url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    kwargs = {}
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in _IN_MEMORY_URLS:
            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine):
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine):
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(engine)


def drop_db(engine):
    """Drop every table (used by the seed script and tests)."""
    Base.metadata.drop_all(engine)


@contextmanager
def session_scope(session_factory):
    """Provide a transactional scope: commit on success, else roll back."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
