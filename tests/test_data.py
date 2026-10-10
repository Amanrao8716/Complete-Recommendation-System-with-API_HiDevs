"""Hand-crafted sample dataset (14 users, 24 content items) for tests."""

WEB, DATA, ML, CLOUD = (
    "Web Development",
    "Data Science",
    "Machine Learning",
    "Cloud & DevOps",
)

# (title, category, difficulty, popularity, [skills]) -> ids 1..24
_CONTENT = [
    ("HTML Basics", WEB, 1, 90, ["HTML", "CSS"]),
    ("CSS Layouts", WEB, 2, 70, ["CSS", "HTML"]),
    ("JavaScript Essentials", WEB, 2, 95, ["JavaScript"]),
    ("React Fundamentals", WEB, 3, 88, ["React", "JavaScript"]),
    ("Advanced React Patterns", WEB, 5, 40, ["React", "JavaScript"]),
    ("Full-Stack Web Project", WEB, 4, 60, ["JavaScript", "React", "HTML"]),
    ("Python for Beginners", DATA, 1, 98, ["Python"]),
    ("SQL Fundamentals", DATA, 2, 85, ["SQL"]),
    ("Pandas in Practice", DATA, 3, 75, ["Pandas", "Python"]),
    ("Statistics for Data Science", DATA, 3, 55, ["Statistics", "Python"]),
    ("Advanced SQL Analytics", DATA, 4, 45, ["SQL", "Statistics"]),
    ("Data Analysis Capstone", DATA, 4, 50, ["Pandas", "SQL", "Statistics"]),
    ("Intro to Machine Learning", ML, 2, 92, ["Machine Learning", "Python"]),
    ("PyTorch Basics", ML, 3, 70, ["PyTorch", "Deep Learning"]),
    ("NLP Foundations", ML, 3, 65, ["NLP", "Machine Learning"]),
    ("Deep Learning Specialization", ML, 4, 80, ["Deep Learning", "PyTorch"]),
    ("Transformers in Depth", ML, 5, 60, ["NLP", "Deep Learning"]),
    ("ML Project Studio", ML, 4, 45, ["Machine Learning", "PyTorch", "NLP"]),
    ("Linux Command Line", CLOUD, 1, 80, ["Linux"]),
    ("Docker Essentials", CLOUD, 2, 90, ["Docker", "Linux"]),
    ("AWS Cloud Practitioner", CLOUD, 2, 85, ["AWS"]),
    ("CI/CD Pipelines", CLOUD, 3, 60, ["CI/CD", "Docker"]),
    ("AWS Solutions Deep Dive", CLOUD, 4, 50, ["AWS", "Docker", "Linux"]),
    ("DevOps Capstone", CLOUD, 5, 35, ["CI/CD", "AWS", "Docker"]),
]

# (name, interests, {skill: proficiency}) -> ids 1..14
_USERS = [
    (
        "Asha",
        "web development, javascript",
        {"HTML": 0.8, "CSS": 0.7, "JavaScript": 0.5},
    ),
    (
        "Bharat",
        "web development, react",
        {"HTML": 0.6, "JavaScript": 0.6, "React": 0.3},
    ),
    ("Chitra", "css", {"HTML": 0.4, "CSS": 0.3}),
    ("Dev", "data science, python", {"Python": 0.7, "SQL": 0.4}),
    ("Esha", "data science, sql", {"Python": 0.5, "SQL": 0.6, "Pandas": 0.3}),
    ("Farhan", "statistics", {"Python": 0.3, "Statistics": 0.2}),
    (
        "Gita",
        "machine learning, nlp",
        {"Python": 0.7, "Machine Learning": 0.5},
    ),
    (
        "Harsh",
        "machine learning, deep learning",
        {"Python": 0.8, "Machine Learning": 0.6, "PyTorch": 0.4},
    ),
    ("Ira", "nlp", {"Python": 0.6, "NLP": 0.3}),
    ("Jai", "cloud & devops, docker", {"Linux": 0.6, "Docker": 0.4}),
    ("Kavya", "aws", {"Linux": 0.3, "AWS": 0.3}),
    ("Lokesh", "devops, ci/cd", {"Docker": 0.7, "Linux": 0.7, "CI/CD": 0.4}),
    ("Nina (cold, has interests)", "web development", {}),
    ("Omar (cold, blank)", "", {}),
]

# (user_id, content_id, type, rating)
_INTERACTIONS = [
    (1, 1, "complete", None),
    (1, 2, "complete", None),
    (1, 3, "like", None),
    (1, 4, "view", None),
    (2, 1, "complete", None),
    (2, 3, "complete", None),
    (2, 4, "rate", 5),
    (3, 1, "like", None),
    (3, 2, "view", None),
    (3, 3, "rate", 4),
    (4, 7, "complete", None),
    (4, 8, "complete", None),
    (4, 9, "like", None),
    (5, 7, "complete", None),
    (5, 8, "like", None),
    (5, 9, "rate", 5),
    (5, 10, "view", None),
    (6, 7, "complete", None),
    (6, 10, "rate", 4),
    (6, 1, "dislike", None),
    (7, 13, "complete", None),
    (7, 15, "like", None),
    (7, 14, "rate", 4),
    (7, 7, "complete", None),
    (8, 13, "complete", None),
    (8, 14, "complete", None),
    (8, 16, "like", None),
    (9, 13, "like", None),
    (9, 15, "rate", 5),
    (9, 17, "view", None),
    (10, 19, "complete", None),
    (10, 20, "complete", None),
    (10, 22, "like", None),
    (11, 19, "complete", None),
    (11, 21, "rate", 5),
    (11, 20, "like", None),
    (12, 20, "complete", None),
    (12, 22, "complete", None),
    (12, 23, "rate", 4),
    (12, 3, "dislike", None),
]

SAMPLE_DATASET = {
    "skills": sorted({s for c in _CONTENT for s in c[4]}),
    "contents": [
        {
            "title": t,
            "category": cat,
            "difficulty": d,
            "popularity": p,
            "skills": s,
        }
        for t, cat, d, p, s in _CONTENT
    ],
    "users": [
        {"name": n, "interests": i, "proficiency": p} for n, i, p in _USERS
    ],
    "interactions": [
        {"user": u, "content": c, "type": t, "rating": r}
        for u, c, t, r in _INTERACTIONS
    ],
}


def test_dataset_meets_minimum_size():
    assert len(SAMPLE_DATASET["users"]) >= 10
    assert len(SAMPLE_DATASET["contents"]) >= 20


def test_dataset_references_are_valid():
    n_users = len(SAMPLE_DATASET["users"])
    n_content = len(SAMPLE_DATASET["contents"])
    for row in SAMPLE_DATASET["interactions"]:
        assert 1 <= row["user"] <= n_users
        assert 1 <= row["content"] <= n_content
