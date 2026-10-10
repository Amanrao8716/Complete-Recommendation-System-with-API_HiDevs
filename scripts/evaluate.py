"""Offline evaluation: precision@5, recall@5, NDCG@5 + report + charts.

Protocol
--------
* 30% of every eligible user's positive interactions are held out.
* The system sees only the remaining interactions.
* Recommendations (excluding already-seen items) are scored against the
  held-out items.
* A feedback experiment starts "probe" users from interests only and
  feeds them interactions one by one through ``record_feedback``.

Run::

    python scripts/evaluate.py            # writes evaluation_report.md
"""

import argparse
import copy
import os
import random
import sys
from collections import defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LOG_LEVEL", "WARNING")

from data.database import (  # noqa: E402
    init_db,
    make_engine,
    make_session_factory,
)
from engine.domain import (  # noqa: E402
    POSITIVE_THRESHOLD,
    interaction_strength,
)
from engine.evaluator import catalog_coverage, evaluate_rankings  # noqa: E402
from engine.orchestrator import RecommendationOrchestrator  # noqa: E402
from scripts.load_test import run_in_process  # noqa: E402
from scripts.seed_data import generate_dataset, seed_database  # noqa: E402

K = 5
STRATEGIES = ["hybrid", "skill_based", "collaborative", "popularity"]
FEEDBACK_STEPS = [0, 1, 2, 3, 5, 8]


def _is_positive(item):
    strength = interaction_strength(item["type"], item.get("rating"))
    return strength >= POSITIVE_THRESHOLD


def split_dataset(dataset, test_fraction=0.3, seed=7, min_positives=4):
    """Hold out positives; return ``(train_dataset, relevant_by_user)``."""
    rng = random.Random(seed)
    positives = defaultdict(list)
    for index, item in enumerate(dataset["interactions"]):
        if _is_positive(item):
            positives[item["user"]].append(index)
    held, relevant = set(), {}
    for user, indexes in sorted(positives.items()):
        if len(indexes) < min_positives:
            continue
        test = rng.sample(indexes, max(1, round(test_fraction * len(indexes))))
        held.update(test)
        relevant[user] = {dataset["interactions"][i]["content"] for i in test}
    train = dict(dataset)
    train["interactions"] = [
        it for i, it in enumerate(dataset["interactions"]) if i not in held
    ]
    return train, relevant


def build_orchestrator(dataset):
    """Create an in-memory database + orchestrator for ``dataset``."""
    engine = make_engine("sqlite://")
    init_db(engine)
    factory = make_session_factory(engine)
    seed_database(factory, dataset)
    return RecommendationOrchestrator(factory, cache_ttl=0)


def _top_ids(orchestrator, user, strategy, k=K):
    result = orchestrator.get_recommendations_detailed(
        user, k, strategy, use_cache=False
    )
    return [item.content_id for item in result.items]


def evaluate_strategies(train, relevant, k=K, random_draws=30, seed=3):
    """Return ``{strategy: metrics}`` including a random baseline."""
    orchestrator = build_orchestrator(train)
    n_content = len(train["contents"])
    results = {}
    for strategy in STRATEGIES:
        recs = {u: _top_ids(orchestrator, u, strategy, k) for u in relevant}
        metrics = evaluate_rankings(recs, relevant, k)
        metrics["coverage"] = catalog_coverage(recs, n_content)
        results[strategy] = metrics

    seen = defaultdict(set)
    for item in train["interactions"]:
        seen[item["user"]].add(item["content"])
    rng = random.Random(seed)
    sums = defaultdict(float)
    for _ in range(random_draws):
        recs = {}
        for user in relevant:
            pool = [c for c in range(1, n_content + 1) if c not in seen[user]]
            recs[user] = rng.sample(pool, min(k, len(pool)))
        metrics = evaluate_rankings(recs, relevant, k)
        for name in ("precision", "recall", "ndcg"):
            sums[name] += metrics[name] / random_draws
    results["random"] = {
        **dict(sums),
        "users": len(relevant),
        "coverage": float("nan"),
    }
    return results


def feedback_curve(train, relevant, steps=FEEDBACK_STEPS, k=K, seed=11):
    """Measure quality as an interests-only user supplies feedback."""
    by_user = defaultdict(list)
    for item in train["interactions"]:
        by_user[item["user"]].append(item)
    feed = {
        u: [it for it in by_user[u] if _is_positive(it)]
        for u in relevant
        if sum(_is_positive(it) for it in by_user[u]) >= max(steps)
    }
    rng = random.Random(seed)
    probes = sorted(rng.sample(sorted(feed), min(40, len(feed))))

    probe_data = copy.deepcopy(train)
    for user in probes:
        probe_data["users"][user - 1]["proficiency"] = {}
    probe_set = set(probes)
    probe_data["interactions"] = [
        it for it in train["interactions"] if it["user"] not in probe_set
    ]
    orchestrator = build_orchestrator(probe_data)

    rows, fed = [], 0
    for step in steps:
        for user in probes:
            for item in feed[user][fed:step]:
                orchestrator.record_feedback(
                    user, item["content"], item["type"], item.get("rating")
                )
        fed = step
        recs = {u: _top_ids(orchestrator, u, "auto", k) for u in probes}
        metrics = evaluate_rankings(recs, {u: relevant[u] for u in probes}, k)
        metrics["feedback_items"] = step
        rows.append(metrics)
    return rows, len(probes)


def _fmt(value):
    return "-" if value != value else f"{value:.3f}"


def make_charts(strategy_results, curve_rows, out_dir):
    """Write PNG charts; return the list of file names (may be empty)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    names = list(strategy_results)
    metrics = [
        ("precision", "Precision@5"),
        ("recall", "Recall@5"),
        ("ndcg", "NDCG@5"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for offset, (key, label) in enumerate(metrics):
        xs = [i + (offset - 1) * width for i in range(len(names))]
        ax.bar(
            xs, [strategy_results[n][key] for n in names], width, label=label
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_title("Recommendation quality by strategy (held-out positives)")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "strategy_comparison.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r["feedback_items"] for r in curve_rows]
    for key, label in (("precision", "Precision@5"), ("ndcg", "NDCG@5")):
        ax.plot(xs, [r[key] for r in curve_rows], marker="o", label=label)
    ax.set_xlabel("feedback items recorded per user")
    ax.set_ylabel("score")
    ax.set_title("Does feedback improve results? (interests-only start)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feedback_curve.png"), dpi=130)
    plt.close(fig)
    return ["strategy_comparison.png", "feedback_curve.png"]


def write_report(
    path, info, strategy_results, curve_rows, n_probes, load, charts
):
    """Write ``evaluation_report.md``."""
    lat = load["latency_ms"]
    checks = [
        ("Response time p95 < 500 ms", lat["p95"] < 500, f"{lat['p95']} ms"),
        (
            "Throughput >= 10 req/s",
            load["throughput_rps"] >= 10,
            f"{load['throughput_rps']} req/s",
        ),
        (
            "No failed requests",
            load["errors"] == 0,
            f"{load['errors']} errors",
        ),
        (
            "Hybrid beats popularity (NDCG@5)",
            strategy_results["hybrid"]["ndcg"]
            > strategy_results["popularity"]["ndcg"],
            f"{strategy_results['hybrid']['ndcg']:.3f} vs "
            f"{strategy_results['popularity']['ndcg']:.3f}",
        ),
        (
            "Hybrid beats random (NDCG@5)",
            strategy_results["hybrid"]["ndcg"]
            > strategy_results["random"]["ndcg"],
            f"{strategy_results['hybrid']['ndcg']:.3f} vs "
            f"{strategy_results['random']['ndcg']:.3f}",
        ),
        (
            "Feedback improves precision@5",
            curve_rows[-1]["precision"] > curve_rows[0]["precision"],
            f"{curve_rows[0]['precision']:.3f} -> "
            f"{curve_rows[-1]['precision']:.3f}",
        ),
    ]
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated {date.today().isoformat()} by `scripts/evaluate.py`._",
        "",
        "## 1. Summary",
        "",
        "| Check | Result | Value |",
        "|---|---|---|",
    ]
    for name, ok, value in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {value} |")

    lines += [
        "",
        "## 2. Dataset and protocol",
        "",
        f"- Synthetic dataset: **{info['users']} users**, "
        f"**{info['content']} content items**, "
        f"{info['interactions']} interactions (seeded, reproducible).",
        f"- Evaluated users: **{strategy_results['hybrid']['users']}** "
        "(users with at least 4 positive interactions).",
        "- 30% of each user's positive interactions (like, complete, "
        "rating >= 4) are held out; the engine only sees the rest.",
        f"- Metrics are computed at k = {K}. Items a user already interacted "
        "with are never recommended.",
        "",
        "## 3. Strategy comparison",
        "",
        "| Strategy | Precision@5 | Recall@5 | NDCG@5 | Catalog coverage |",
        "|---|---|---|---|---|",
    ]
    for name, m in strategy_results.items():
        lines.append(
            f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
            f"{_fmt(m['ndcg'])} | {_fmt(m['coverage'])} |"
        )
    lines.append("")
    lines.append("`random` is the average of 30 random draws (lower bound).")
    if charts:
        lines += [
            "",
            "![Strategy comparison](reports/strategy_comparison.png)",
        ]

    lines += [
        "",
        "## 4. Does feedback improve results?",
        "",
        f"{n_probes} probe users start with **interests only** (no history, "
        "no skills). Their real interactions are fed one at a time through "
        "`record_feedback`, and recommendations are scored against "
        "held-out items.",
        "",
        "| Feedback items | Strategy path | Precision@5 | Recall@5 | NDCG@5 |",
        "|---|---|---|---|---|",
    ]
    for row in curve_rows:
        path_name = "cold_start" if row["feedback_items"] == 0 else "hybrid"
        lines.append(
            f"| {row['feedback_items']} | {path_name} "
            f"| {row['precision']:.3f} "
            f"| {row['recall']:.3f} | {row['ndcg']:.3f} |"
        )
    if charts:
        lines += ["", "![Feedback curve](reports/feedback_curve.png)"]

    lines += [
        "",
        "## 5. Load test (10 concurrent simulated users)",
        "",
        "In-process run of the Flask app (90% `GET /recommend`, "
        "10% `POST /feedback`).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Concurrent workers | {load['workers']} |",
        f"| Total requests | {load['total_requests']} |",
        f"| Errors | {load['errors']} |",
        f"| Throughput | {load['throughput_rps']} req/s |",
        f"| Latency avg / p50 | {lat['avg']} / {lat['p50']} ms |",
        f"| Latency p95 / p99 / max | {lat['p95']} / {lat['p99']} "
        f"/ {lat['max']} ms |",
        f"| Cache hit rate | {load['cache_hit_rate']:.1%} |",
        "",
        "## 6. Reproduce",
        "",
        "```bash",
        "python scripts/evaluate.py",
        "python scripts/load_test.py --workers 10 --requests 100",
        "pytest --cov=data --cov=engine --cov=api",
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--content", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "evaluation_report.md")
    )
    args = parser.parse_args(argv)

    dataset = generate_dataset(args.users, args.content, args.seed)
    info = {
        "users": len(dataset["users"]),
        "content": len(dataset["contents"]),
        "interactions": len(dataset["interactions"]),
    }
    train, relevant = split_dataset(dataset)
    strategy_results = evaluate_strategies(train, relevant)
    curve_rows, n_probes = feedback_curve(train, relevant)
    load = run_in_process(workers=10, requests_per_worker=100)
    charts = make_charts(
        strategy_results, curve_rows, os.path.join(ROOT, "reports")
    )
    write_report(
        args.output, info, strategy_results, curve_rows, n_probes, load, charts
    )

    print(f"{'strategy':<14}{'P@5':>8}{'R@5':>8}{'NDCG@5':>8}")
    for name, m in strategy_results.items():
        print(
            f"{name:<14}"
            f"{m['precision']:>8.3f}{m['recall']:>8.3f}"
            f"{m['ndcg']:>8.3f}"
        )
    print(
        "feedback curve (P@5):", [round(r["precision"], 3) for r in curve_rows]
    )
    print("report written to", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
