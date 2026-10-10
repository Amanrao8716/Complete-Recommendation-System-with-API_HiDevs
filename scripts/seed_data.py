"""Populate the database with a deterministic synthetic dataset.

Examples::

    python scripts/seed_data.py --reset
    python scripts/seed_data.py --reset --users 150 --content 100
    python scripts/seed_data.py --small        # the hand-made test dataset
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.database import (  # noqa: E402
    drop_db,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from data.repositories import (  # noqa: E402
    ContentRepository,
    ContentSkillRepository,
    InteractionRepository,
    SkillRepository,
    UserRepository,
    UserSkillRepository,
)

TRACKS = {
    "Web Development": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "REST APIs",
    ],
    "Data Science": [
        "Python",
        "Pandas",
        "SQL",
        "Statistics",
        "Data Visualization",
        "NumPy",
    ],
    "Machine Learning": [
        "Machine Learning",
        "Deep Learning",
        "PyTorch",
        "NLP",
        "Model Evaluation",
        "Feature Engineering",
    ],
    "Cloud & DevOps": [
        "AWS",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "Linux",
        "Terraform",
    ],
    "Software Engineering": [
        "Algorithms",
        "Data Structures",
        "System Design",
        "Git",
        "Testing",
        "Databases",
    ],
}
KINDS = [
    "Fundamentals",
    "Crash Course",
    "in Practice",
    "Deep Dive",
    "Masterclass",
    "Workshop",
    "Project Lab",
    "Bootcamp",
]
FIRST_NAMES = [
    "Aarav",
    "Diya",
    "Kabir",
    "Meera",
    "Rohan",
    "Isha",
    "Vihaan",
    "Anaya",
    "Arjun",
    "Saanvi",
    "Reyansh",
    "Myra",
    "Ishaan",
    "Kiara",
    "Advait",
    "Tara",
    "Neel",
    "Riya",
    "Yash",
    "Zoya",
]
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def generate_dataset(n_users=40, n_content=60, seed=42):
    """Create a reproducible dataset with learnable structure.

    Users belong to one or two skill tracks and mostly interact with
    content from those tracks at a difficulty near their level.  Two
    extra cold-start users (no history) are appended at the end.
    """
    rng = random.Random(seed)
    track_names = list(TRACKS)
    all_skills = [s for skills in TRACKS.values() for s in skills]

    contents, used_titles = [], set()
    per_track = max(1, n_content // len(track_names))
    for track in track_names:
        for _ in range(per_track):
            chosen = rng.sample(TRACKS[track], rng.choice([1, 2, 3]))
            if rng.random() < 0.3:
                other = rng.choice([t for t in track_names if t != track])
                chosen.append(rng.choice(TRACKS[other]))
            kind = rng.choice(KINDS)
            title = f"{chosen[0]} {kind}"
            if title in used_titles:
                title = f"{chosen[0]} & {chosen[-1]} {kind}"
            if title in used_titles:
                title = f"{title} #{len(contents) + 1}"
            used_titles.add(title)
            contents.append(
                {
                    "title": title,
                    "category": track,
                    "difficulty": rng.randint(1, 5),
                    "popularity": round(rng.uniform(5, 100), 1),
                    "skills": chosen,
                }
            )
    rng.shuffle(contents)

    users = []
    for i in range(n_users):
        primary = rng.sample(track_names, 2 if rng.random() < 0.3 else 1)
        base = rng.uniform(0.1, 0.9)
        proficiency = {}
        for track in primary:
            for skill in TRACKS[track]:
                if rng.random() < 0.7:
                    value = base + rng.uniform(-0.2, 0.2)
                    proficiency[skill] = round(max(0.0, min(1.0, value)), 2)
        for skill in rng.sample(all_skills, 2):
            if skill not in proficiency and rng.random() < 0.3:
                proficiency[skill] = round(rng.uniform(0.0, 0.3), 2)
        picks = [s.lower() for s in rng.sample(TRACKS[primary[0]], 2)]
        users.append(
            {
                "name": f"{FIRST_NAMES[i % 20]} {chr(65 + (i // 20) % 26)}.",
                "interests": ", ".join([primary[0].lower()] + picks),
                "proficiency": proficiency,
                "_primary": set(primary),
                "_level": 1 + 4 * base,
            }
        )

    interactions = []
    for user_id, user in enumerate(users, start=1):
        weights = []
        for item in contents:
            on_track = item["category"] in user["_primary"]
            fit = math.exp(
                -abs(item["difficulty"] - (user["_level"] + 0.5)) / 1.5
            )
            weights.append(
                (0.3 + item["popularity"] / 100)
                * (1.0 if on_track else 0.05)
                * fit
            )
        pool = list(range(len(contents)))
        for _ in range(min(rng.randint(8, 20), len(pool))):
            idx = rng.choices(pool, weights=[weights[p] for p in pool])[0]
            pool.remove(idx)
            on_track = contents[idx]["category"] in user["_primary"]
            if on_track:
                probs = [0.20, 0.25, 0.35, 0.18, 0.02]
            else:
                probs = [0.40, 0.10, 0.05, 0.30, 0.15]
            kind = rng.choices(
                ["view", "like", "complete", "rate", "dislike"], probs
            )[0]
            rating = None
            if kind == "rate":
                rating = rng.choice(
                    [3, 4, 4, 5, 5] if on_track else [1, 2, 2, 3]
                )
            interactions.append(
                {
                    "user": user_id,
                    "content": idx + 1,
                    "type": kind,
                    "rating": rating,
                }
            )

    for user in users:
        user.pop("_primary")
        user.pop("_level")
    users.append(
        {
            "name": "Cold Casey",
            "interests": "web development",
            "proficiency": {},
        }
    )
    users.append({"name": "Blank Blake", "interests": "", "proficiency": {}})

    return {
        "skills": all_skills,
        "contents": contents,
        "users": users,
        "interactions": interactions,
    }


def seed_database(session_factory, dataset):
    """Insert a dataset (see ``generate_dataset``) and return counts.

    Users and contents get ids 1..N in list order.
    """
    with session_scope(session_factory) as session:
        skills = SkillRepository(session)
        skill_ids = {n: skills.get_or_create(n).id for n in dataset["skills"]}

        content_repo = ContentRepository(session)
        links = ContentSkillRepository(session)
        for item in dataset["contents"]:
            row = content_repo.create(
                item["title"],
                item["category"],
                item["difficulty"],
                item["popularity"],
            )
            for name in item["skills"]:
                links.link(row.id, skill_ids[name])

        user_repo = UserRepository(session)
        user_skill_repo = UserSkillRepository(session)
        for item in dataset["users"]:
            row = user_repo.create(item["name"], item.get("interests", ""))
            for name, value in item.get("proficiency", {}).items():
                user_skill_repo.set_proficiency(row.id, skill_ids[name], value)

        inter_repo = InteractionRepository(session)
        for offset, item in enumerate(dataset["interactions"]):
            inter_repo.record(
                item["user"],
                item["content"],
                item["type"],
                item.get("rating"),
                BASE_TIME + timedelta(minutes=offset),
            )
    return {
        "users": len(dataset["users"]),
        "content": len(dataset["contents"]),
        "skills": len(dataset["skills"]),
        "interactions": len(dataset["interactions"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", help="database URL (default: DATABASE_URL)")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--content", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small", action="store_true", help="use test dataset"
    )
    parser.add_argument(
        "--reset", action="store_true", help="drop tables first"
    )
    parser.add_argument(
        "--if-empty", action="store_true", help="skip when users already exist"
    )
    args = parser.parse_args(argv)

    engine = make_engine(args.db)
    if args.reset:
        drop_db(engine)
    init_db(engine)
    factory = make_session_factory(engine)

    if args.if_empty:
        with session_scope(factory) as session:
            if UserRepository(session).count():
                print("Database already seeded; nothing to do.")
                return 0

    if args.small:
        from tests.test_data import SAMPLE_DATASET

        dataset = SAMPLE_DATASET
    else:
        dataset = generate_dataset(args.users, args.content, args.seed)
    counts = seed_database(factory, dataset)
    print("Seeded:", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
