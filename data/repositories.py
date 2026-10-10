"""Data access layer: one repository class per table."""

from sqlalchemy import func, select

from data.models import (
    Content,
    ContentSkill,
    Interaction,
    Skill,
    User,
    UserSkill,
)


class BaseRepository:
    """Holds the SQLAlchemy session shared by all repositories."""

    def __init__(self, session):
        self.session = session


class UserRepository(BaseRepository):
    """CRUD helpers for ``users``."""

    def get(self, user_id):
        """Return the user or ``None``."""
        return self.session.get(User, user_id)

    def exists(self, user_id):
        """Return True if the user exists."""
        return self.get(user_id) is not None

    def get_all(self):
        """Return all users ordered by id."""
        return list(self.session.scalars(select(User).order_by(User.id)))

    def create(self, name, interests=""):
        """Insert a user and return it (id is populated)."""
        user = User(name=name, interests=interests)
        self.session.add(user)
        self.session.flush()
        return user

    def count(self):
        """Return the number of users."""
        return self.session.scalar(select(func.count(User.id)))


class ContentRepository(BaseRepository):
    """CRUD helpers for ``content``."""

    def get(self, content_id):
        """Return the content item or ``None``."""
        return self.session.get(Content, content_id)

    def get_all(self):
        """Return all content ordered by id."""
        stmt = select(Content).order_by(Content.id)
        return list(self.session.scalars(stmt))

    def create(self, title, category, difficulty, popularity=0.0):
        """Insert a content item and return it."""
        item = Content(
            title=title,
            category=category,
            difficulty=difficulty,
            popularity=popularity,
        )
        self.session.add(item)
        self.session.flush()
        return item

    def get_by_skills(self, skill_ids):
        """Return content teaching at least one of ``skill_ids``."""
        ids = list(skill_ids)
        if not ids:
            return []
        stmt = (
            select(Content)
            .join(ContentSkill, ContentSkill.content_id == Content.id)
            .where(ContentSkill.skill_id.in_(ids))
            .distinct()
            .order_by(Content.id)
        )
        return list(self.session.scalars(stmt))


class SkillRepository(BaseRepository):
    """CRUD helpers for ``skills``."""

    def get(self, skill_id):
        """Return the skill or ``None``."""
        return self.session.get(Skill, skill_id)

    def get_by_name(self, name):
        """Return the skill with this name or ``None``."""
        stmt = select(Skill).where(Skill.name == name)
        return self.session.scalars(stmt).first()

    def get_or_create(self, name):
        """Return the skill, creating it when missing."""
        skill = self.get_by_name(name)
        if skill is None:
            skill = Skill(name=name)
            self.session.add(skill)
            self.session.flush()
        return skill

    def get_all(self):
        """Return all skills ordered by id."""
        return list(self.session.scalars(select(Skill).order_by(Skill.id)))


class UserSkillRepository(BaseRepository):
    """Helpers for ``user_skills``."""

    def get_for_user(self, user_id):
        """Return ``{skill_id: proficiency}`` for one user."""
        stmt = select(UserSkill).where(UserSkill.user_id == user_id)
        return {r.skill_id: r.proficiency for r in self.session.scalars(stmt)}

    def get_all(self):
        """Return every user-skill row."""
        return list(self.session.scalars(select(UserSkill)))

    def set_proficiency(self, user_id, skill_id, proficiency):
        """Insert or update a proficiency value (clamped to 0..1)."""
        value = max(0.0, min(1.0, float(proficiency)))
        row = self.session.get(UserSkill, (user_id, skill_id))
        if row is None:
            row = UserSkill(
                user_id=user_id, skill_id=skill_id, proficiency=value
            )
            self.session.add(row)
        else:
            row.proficiency = value
        self.session.flush()
        return row


class ContentSkillRepository(BaseRepository):
    """Helpers for ``content_skills``."""

    def link(self, content_id, skill_id):
        """Attach a skill to a content item (idempotent)."""
        if self.session.get(ContentSkill, (content_id, skill_id)) is None:
            self.session.add(
                ContentSkill(content_id=content_id, skill_id=skill_id)
            )
            self.session.flush()

    def get_skill_ids(self, content_id):
        """Return the skill ids taught by one content item."""
        stmt = select(ContentSkill.skill_id).where(
            ContentSkill.content_id == content_id
        )
        return sorted(self.session.scalars(stmt))

    def get_all(self):
        """Return ``[(content_id, skill_id), ...]``."""
        stmt = select(ContentSkill.content_id, ContentSkill.skill_id)
        return [tuple(row) for row in self.session.execute(stmt)]


class InteractionRepository(BaseRepository):
    """Helpers for ``interactions``."""

    def record(self, user_id, content_id, type_, rating=None, created_at=None):
        """Store one interaction and return it."""
        row = Interaction(
            user_id=user_id,
            content_id=content_id,
            type=type_,
            rating=rating,
        )
        if created_at is not None:
            row.created_at = created_at
        self.session.add(row)
        self.session.flush()
        return row

    def get_user_history(self, user_id, limit=None):
        """Return a user's interactions, newest first."""
        stmt = (
            select(Interaction)
            .where(Interaction.user_id == user_id)
            .order_by(Interaction.created_at.desc(), Interaction.id.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def get_all(self):
        """Return all interactions in chronological order."""
        stmt = select(Interaction).order_by(
            Interaction.created_at, Interaction.id
        )
        return list(self.session.scalars(stmt))

    def count_for_user(self, user_id):
        """Return how many interactions a user has."""
        stmt = select(func.count(Interaction.id)).where(
            Interaction.user_id == user_id
        )
        return self.session.scalar(stmt)
