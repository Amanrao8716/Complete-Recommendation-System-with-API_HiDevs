# Day 30 Capstone: Recommendation System

A complete learning-content recommendation service: **SQLite database →
recommendation engine (5 strategies, caching, explanations) → Flask REST
API**, with tests, offline evaluation, a load test, Docker and a Postman
collection.

```
User Request → API → RecommendationOrchestrator → Database (SQLite)
                ↓              ↓
              Metrics     Cache + candidate generation + scoring
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/seed_data.py --reset        # 40 users, 60 content items
python -m api.app                          # http://127.0.0.1:5000

curl "http://127.0.0.1:5000/recommend/1?limit=5"
curl -X POST http://127.0.0.1:5000/feedback \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "content_id": 7, "type": "complete"}'
```

In GitHub Codespaces, forward port 5000 and use the forwarded URL.
`scripts/demo.sh` runs a scripted walkthrough, handy for the demo video.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///recsys.db` | Database location |
| `API_KEY` | _(empty = auth off)_ | If set, `/recommend`, `/feedback`, `/metrics` require header `X-API-Key` |
| `CACHE_TTL` | `300` | Recommendation cache TTL in seconds |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PORT` | `5000` | Server port |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/recommend/<user_id>?limit=5&strategy=auto` | Recommendations (limit 1-50). Strategies: `auto`, `hybrid`, `skill_based`, `collaborative`, `popularity`, `cold_start` |
| POST | `/feedback` | Body: `{"user_id", "content_id", "type", "rating"?}`; `type` is `view`, `like`, `complete`, `rate` (needs `rating`) or `dislike`; `rating` 1-5 |
| GET | `/health` | Liveness + DB check |
| GET | `/metrics` | Request counts, latency percentiles per endpoint, cache hit rate |

Every response carries an `X-Request-ID` header (a client-supplied one is
echoed back) which also appears in the logs and in error bodies.

Errors are JSON: `{"error": {"code", "message", "request_id"}}` with
`400` (bad input), `401` (bad API key), `404` (unknown user/content/route),
`405`, `500`.

Example response:

```json
{
  "request_id": "9f2c1a7d03be", "user_id": 1, "strategy": "hybrid",
  "count": 1, "cached": false, "latency_ms": 1.4,
  "recommendations": [{
    "content_id": 6, "title": "Full-Stack Web Project",
    "category": "Web Development", "difficulty": 4, "score": 0.4982,
    "strategy": "hybrid",
    "explanation": "Recommended because it fits your interest in Web Development and builds skills you care about (JavaScript, HTML and React).",
    "components": {"skill_match": 0.62, "category_affinity": 0.6, "...": 0}
  }]
}
```

## How the engine works

1. **Candidate generation** (`engine/candidate_gen.py`): items sharing skills
   the user needs, items in categories they like, items liked by similar
   users (cosine similarity on interaction vectors), popular items.
   Already-seen items are removed; popularity backfills if too few remain.
2. **Scoring** (`engine/scorer.py`): five scorers (skill match, category
   affinity, difficulty fit, collaborative, popularity) combined with
   per-strategy weights, then a light category-diversity re-rank.
3. **Strategies**: `hybrid` (default for users with history),
   `skill_based`, `collaborative`, `popularity`, `cold_start`.
4. **Cold start**: users with no history and no skills get `cold_start`:
   interests (if any) + popularity + beginner-friendly difficulty.
5. **Feedback loop**: `POST /feedback` stores the interaction, updates the
   in-memory model immediately, invalidates that user's cache and, for
   `complete`, raises proficiency in the content's skills.
6. **Explanations**: built from the two largest weighted score components.
7. **Cache**: in-memory dict with TTL, per (user, limit, strategy), cleared
   per user on feedback.

## Project layout

```
data/      database.py, models.py (6 tables), repositories.py
engine/    orchestrator.py, similarity.py, candidate_gen.py, scorer.py,
           evaluator.py, domain.py
api/       app.py
tests/     test_data.py (sample dataset), test_repositories.py,
           test_engine.py, test_api.py, test_scripts.py
scripts/   seed_data.py, evaluate.py, load_test.py, demo.sh
```

> The `similarity`, `candidate_gen`, `scorer` and `evaluator` modules are the
> "Day 29" components. If you have your own Day 29 versions, keep the public
> interfaces used by `orchestrator.py` (`generate(ctx, strategy, ...)`,
> `WeightedScorer.score(...)`, `evaluate_rankings(...)`).

## Tests, evaluation, load test

```bash
pytest --cov=data --cov=engine --cov=api --cov-report=term-missing
python scripts/evaluate.py                       # writes evaluation_report.md
python scripts/load_test.py --workers 10 --requests 100
python scripts/load_test.py --url http://127.0.0.1:5000   # live server
flake8 .
```

The test dataset (`tests/test_data.py`) has 14 users and 24 content items.
`evaluation_report.md` contains precision@5 / recall@5 / NDCG@5 per
strategy, a feedback-learning curve, charts and load-test results.

## Docker

```bash
docker build -t recsys .
docker run -p 5000:5000 -e API_KEY=secret recsys
curl -H "X-API-Key: secret" localhost:5000/recommend/1
```

The container seeds the database on first start and serves with gunicorn.
For Render/Heroku-style deployment, use the Dockerfile and set `PORT`.

## Known limitations

- SQLite and an in-process cache suit a single instance; for several
  instances use Postgres and Redis.
- Recommendations for other users are not invalidated when one user gives
  feedback (their cached lists expire after `CACHE_TTL`).
- The evaluation dataset is synthetic; absolute metric values are only
  meaningful relative to the baselines.
