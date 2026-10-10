"""Simple load test simulating concurrent users.

Default mode runs the Flask app in-process (no network).  Use ``--url``
to hit a running server instead::

    python scripts/load_test.py --workers 10 --requests 100
    python scripts/load_test.py --url http://127.0.0.1:5000
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.app import create_app  # noqa: E402
from data.database import (  # noqa: E402
    init_db,
    make_engine,
    make_session_factory,
)
from engine.orchestrator import RecommendationOrchestrator  # noqa: E402
from scripts.seed_data import generate_dataset, seed_database  # noqa: E402

os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep output readable
FEEDBACK_TYPES = ["view", "like", "complete"]


class InProcessTransport:
    """Call the WSGI app directly through Flask's test client."""

    def __init__(self, app, headers=None):
        self.client = app.test_client()
        self.headers = headers or {}

    def request(self, method, path, payload=None):
        start = time.perf_counter()
        response = self.client.open(
            path, method=method, json=payload, headers=self.headers
        )
        elapsed = (time.perf_counter() - start) * 1000
        return response.status_code, elapsed, response.get_json()


class HttpTransport:
    """Call a live server over HTTP."""

    def __init__(self, base_url, headers=None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        self.headers.update(headers or {})

    def request(self, method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self.headers,
            method=method,
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, body = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        elapsed = (time.perf_counter() - start) * 1000
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        return status, elapsed, parsed


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(round(pct / 100 * (len(ordered) - 1)))]


def run_load_test(
    transport_factory,
    n_users,
    n_content,
    workers=10,
    requests_per_worker=100,
    feedback_ratio=0.1,
    seed=1,
):
    """Run ``workers`` concurrent simulated users and summarise the results."""
    latencies, errors = [], []
    lock = threading.Lock()

    def worker(worker_id):
        rng = random.Random(seed * 1000 + worker_id)
        transport = transport_factory()
        local = []
        for _ in range(requests_per_worker):
            user = rng.randint(1, n_users)
            if rng.random() < feedback_ratio:
                payload = {
                    "user_id": user,
                    "content_id": rng.randint(1, n_content),
                    "type": rng.choice(FEEDBACK_TYPES),
                }
                status, ms, _ = transport.request("POST", "/feedback", payload)
                ok = status == 201
            else:
                path = f"/recommend/{user}?limit=5"
                status, ms, _ = transport.request("GET", path)
                ok = status == 200
            local.append(ms)
            if not ok:
                with lock:
                    errors.append(status)
        with lock:
            latencies.extend(local)

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(workers)
    ]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    duration = time.perf_counter() - started

    _, _, metrics = transport_factory().request("GET", "/metrics")
    cache = (metrics or {}).get("recommender", {}).get("cache", {})
    total = len(latencies)
    return {
        "workers": workers,
        "total_requests": total,
        "errors": len(errors),
        "duration_seconds": round(duration, 3),
        "throughput_rps": round(total / duration, 1) if duration else 0.0,
        "latency_ms": {
            "avg": round(sum(latencies) / total, 2) if total else 0.0,
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "p99": round(_percentile(latencies, 99), 2),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "cache_hit_rate": cache.get("hit_rate", 0.0),
    }


def run_in_process(
    n_users=40, n_content=60, workers=10, requests_per_worker=100, seed=1
):
    """Build a fresh seeded app and load-test it in-process."""
    dataset = generate_dataset(n_users, n_content, seed=42)
    engine = make_engine("sqlite://")
    init_db(engine)
    factory = make_session_factory(engine)
    seed_database(factory, dataset)
    app = create_app(
        orchestrator=RecommendationOrchestrator(factory, cache_ttl=300),
        api_key="",
    )
    return run_load_test(
        lambda: InProcessTransport(app),
        len(dataset["users"]),
        len(dataset["contents"]),
        workers,
        requests_per_worker,
        seed=seed,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", help="base URL of a running server")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100, help="per worker")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--content", type=int, default=60)
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    args = parser.parse_args(argv)

    if args.url:
        headers = {"X-API-Key": args.api_key} if args.api_key else {}
        result = run_load_test(
            lambda: HttpTransport(args.url, headers),
            args.users,
            args.content,
            args.workers,
            args.requests,
        )
    else:
        result = run_in_process(
            args.users, args.content, args.workers, args.requests
        )
    print(json.dumps(result, indent=2))
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
