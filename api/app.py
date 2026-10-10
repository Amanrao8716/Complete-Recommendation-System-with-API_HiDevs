"""Flask REST API serving recommendations.

Run locally::

    python -m api.app            # development server
    gunicorn 'api.app:create_app()'   # production-style
"""

import hmac
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

from flask import Flask, g, jsonify, render_template, request
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from data.database import (
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from engine.orchestrator import (
    MAX_LIMIT,
    VALID_STRATEGIES,
    ContentNotFoundError,
    InvalidFeedbackError,
    InvalidRequestError,
    RecommendationOrchestrator,
    UserNotFoundError,
)

logger = logging.getLogger("recsys.api")
PROTECTED_ENDPOINTS = {"recommend", "feedback", "metrics"}


class ApiError(Exception):
    """Exception rendered as a JSON error response."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    index = int(round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


class MetricsRegistry:
    """Thread-safe request counters and latency windows."""

    def __init__(self, window=2000):
        self._lock = threading.Lock()
        self._started = time.time()
        self._count = defaultdict(int)
        self._total_ms = defaultdict(float)
        self._max_ms = defaultdict(float)
        self._window = defaultdict(lambda: deque(maxlen=window))
        self._status = defaultdict(int)

    def record(self, endpoint, status, duration_ms):
        """Record one finished request."""
        with self._lock:
            self._count[endpoint] += 1
            self._total_ms[endpoint] += duration_ms
            self._max_ms[endpoint] = max(self._max_ms[endpoint], duration_ms)
            self._window[endpoint].append(duration_ms)
            self._status[str(status)] += 1

    def snapshot(self):
        """Return a JSON-serialisable summary."""
        with self._lock:
            endpoints = {}
            for name, count in self._count.items():
                values = sorted(self._window[name])
                endpoints[name] = {
                    "count": count,
                    "avg_ms": round(self._total_ms[name] / count, 3),
                    "p50_ms": round(_percentile(values, 50), 3),
                    "p95_ms": round(_percentile(values, 95), 3),
                    "max_ms": round(self._max_ms[name], 3),
                }
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "total_requests": sum(self._count.values()),
                "status_codes": dict(self._status),
                "endpoints": endpoints,
            }


def _parse_int(value, name):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_request", f"{name} must be an integer")


def _require_int(payload, name):
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(400, "invalid_request", f"{name} must be an integer")
    return value


def create_app(database_url=None, api_key=None, orchestrator=None, cache_ttl=None):
    """Application factory."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    app = Flask(__name__)
    if orchestrator is None:
        engine = make_engine(database_url)
        init_db(engine)
        ttl = cache_ttl
        if ttl is None:
            ttl = float(os.environ.get("CACHE_TTL", "300"))
        orchestrator = RecommendationOrchestrator(
            make_session_factory(engine), cache_ttl=ttl
        )
    if api_key is None:
        api_key = os.environ.get("API_KEY", "")

    metrics = MetricsRegistry()
    app.extensions["orchestrator"] = orchestrator
    app.extensions["metrics"] = metrics

    # ------------------------------------------------------------------
    # Request lifecycle: tracing, auth, logging, metrics
    # ------------------------------------------------------------------
    @app.before_request
    def start_request():
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming[:64] if incoming else uuid.uuid4().hex[:12]
        g.start = time.perf_counter()
        if api_key and request.endpoint in PROTECTED_ENDPOINTS:
            supplied = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(supplied, api_key):
                raise ApiError(401, "unauthorized", "Missing or invalid API key")

    @app.after_request
    def finish_request(response):
        duration_ms = (time.perf_counter() - g.get("start", time.perf_counter())) * 1000
        rule = request.url_rule.rule if request.url_rule else "unmatched"
        response.headers["X-Request-ID"] = g.get("request_id", "-")
        metrics.record(rule, response.status_code, duration_ms)
        logger.info(
            "request_id=%s %s %s -> %s %.1fms",
            g.get("request_id", "-"),
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        return response

    def error_response(status, code, message):
        body = {
            "error": {
                "code": code,
                "message": message,
                "request_id": g.get("request_id"),
            }
        }
        return jsonify(body), status

    @app.errorhandler(ApiError)
    def handle_api_error(exc):
        return error_response(exc.status, exc.code, exc.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return error_response(
            exc.code, exc.name.lower().replace(" ", "_"), exc.description
        )

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("request_id=%s unhandled error", g.get("request_id"))
        return error_response(500, "internal_error", "Internal server error")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        accept = request.headers.get("Accept", "")
        wants_json = (
            request.args.get("format") == "json"
            or ("application/json" in accept and "text/html" not in accept)
            or ("text/html" not in accept and request.args.get("format") != "html")
        )
        if wants_json:
            return jsonify(
                {
                    "service": "Learning Content Recommendation System API",
                    "status": "online",
                    "request_id": g.request_id,
                    "endpoints": {
                        "health": "/health",
                        "recommend": "/recommend/<user_id>?limit=5&strategy=auto",
                        "feedback": "/feedback",
                        "metrics": "/metrics",
                    },
                    "strategies": sorted(VALID_STRATEGIES),
                    "documentation": {
                        "interactive_ui": "/",
                        "sample_recommendation": "/recommend/1?limit=5",
                        "metrics_endpoint": "/metrics",
                    },
                }
            )
        return render_template("index.html")

    @app.get("/health")
    def health():
        try:
            with session_scope(orchestrator.session_factory) as session:
                session.execute(text("SELECT 1"))
            database, status = "ok", 200
        except Exception:
            logger.exception("health check failed")
            database, status = "error", 503
        stats = orchestrator.stats()
        body = {
            "status": "ok" if status == 200 else "degraded",
            "database": database,
            "users": stats["users"],
            "content_items": stats["content_items"],
            "request_id": g.request_id,
        }
        return jsonify(body), status

    @app.get("/metrics", endpoint="metrics")
    def metrics_endpoint():
        body = metrics.snapshot()
        body["recommender"] = orchestrator.stats()
        return jsonify(body)

    @app.get("/recommend/<user_id>")
    def recommend(user_id):
        uid = _parse_int(user_id, "user_id")
        limit = _parse_int(request.args.get("limit", "5"), "limit")
        strategy = request.args.get("strategy", "auto")
        if not 1 <= limit <= MAX_LIMIT:
            raise ApiError(
                400,
                "invalid_request",
                f"limit must be between 1 and {MAX_LIMIT}",
            )
        if strategy not in VALID_STRATEGIES:
            raise ApiError(
                400,
                "invalid_request",
                f"strategy must be one of {', '.join(VALID_STRATEGIES)}",
            )
        try:
            result = orchestrator.get_recommendations_detailed(uid, limit, strategy)
        except UserNotFoundError as exc:
            raise ApiError(404, "user_not_found", str(exc))
        except InvalidRequestError as exc:
            raise ApiError(400, "invalid_request", str(exc))
        latency = (time.perf_counter() - g.start) * 1000
        return jsonify(
            {
                "request_id": g.request_id,
                "user_id": uid,
                "strategy": result.strategy,
                "count": len(result.items),
                "cached": result.cached,
                "latency_ms": round(latency, 2),
                "recommendations": [i.to_dict() for i in result.items],
            }
        )

    @app.post("/feedback")
    def feedback():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError(400, "invalid_request", "Request body must be a JSON object")
        missing = [f for f in ("user_id", "content_id", "type") if f not in payload]
        if missing:
            raise ApiError(
                400, "invalid_request", f"Missing fields: {', '.join(missing)}"
            )
        user_id = _require_int(payload, "user_id")
        content_id = _require_int(payload, "content_id")
        type_ = payload["type"]
        if not isinstance(type_, str):
            raise ApiError(400, "invalid_request", "type must be a string")
        try:
            record = orchestrator.record_feedback(
                user_id, content_id, type_, payload.get("rating")
            )
        except (UserNotFoundError, ContentNotFoundError) as exc:
            raise ApiError(404, "not_found", str(exc))
        except InvalidFeedbackError as exc:
            raise ApiError(400, "invalid_request", str(exc))
        return (
            jsonify(
                {
                    "request_id": g.request_id,
                    "status": "recorded",
                    "interaction": record,
                }
            ),
            201,
        )

    return app


if __name__ == "__main__":
    create_app().run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
    )
