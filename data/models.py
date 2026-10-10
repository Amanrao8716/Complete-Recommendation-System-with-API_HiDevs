"""SQLAlchemy ORM models (the six tables of the recommendation system)."""

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

INTERACTION_TYPES = ("view", "like", "complete", "rate", "dislike")


def utcnow():
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base class for all models."""


class User(Base):
    """A learner. ``interests`` is a comma separated string."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    interests: Mapped[str] = mapped_column(
        String(500), default="", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    def interest_list(self):
        """Return lower-cased interest tokens."""
        raw = self.interests or ""
        return [t.strip().lower() for t in raw.split(",") if t.strip()]


class Content(Base):
    """A learning item (course, tutorial, project...)."""

    __tablename__ = "content"
    __table_args__ = (
        CheckConstraint(
            "difficulty BETWEEN 1 AND 5", name="ck_content_difficulty"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False)
    popularity: Mapped[float] = mapped_column(Float, default=0.0)


class Skill(Base):
    """A skill that content teaches and users possess."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)


class UserSkill(Base):
    """User proficiency (0..1) in a skill."""

    __tablename__ = "user_skills"
    __table_args__ = (
        CheckConstraint(
            "proficiency BETWEEN 0 AND 1", name="ck_user_skill_proficiency"
        ),
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(
        ForeignKey("skills.id"), primary_key=True
    )
    proficiency: Mapped[float] = mapped_column(Float, default=0.0)


class ContentSkill(Base):
    """Association between content and the skills it teaches."""

    __tablename__ = "content_skills"

    content_id: Mapped[int] = mapped_column(
        ForeignKey("content.id"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(
        ForeignKey("skills.id"), primary_key=True
    )


class Interaction(Base):
    """A user event on a content item (view, like, rating...)."""

    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint(
            "type IN ('view','like','complete','rate','dislike')",
            name="ck_interaction_type",
        ),
        CheckConstraint(
            "rating IS NULL OR rating BETWEEN 1 AND 5",
            name="ck_interaction_rating",
        ),
        Index("ix_interactions_user_time", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content_id: Mapped[int] = mapped_column(
        ForeignKey("content.id"), index=True
    )
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
