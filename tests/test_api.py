"""Tests for the REST API."""

from api.app import create_app


def _post(client, payload, **kw):
    return client.post("/feedback", json=payload, **kw)


def test_health(client):
    response = client.get("/health")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["users"] == 14 and body["content_items"] == 24


def test_index_json(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "online"
    assert "/health" in body["endpoints"]["health"]


def test_index_html(client):
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.content_type
    assert b"Recommendation Engine" in response.data


def test_recommend_success(client):
    response = client.get("/recommend/1?limit=3")
    body = response.get_json()
    assert response.status_code == 200
    assert body["count"] == 3 and len(body["recommendations"]) == 3
    assert body["user_id"] == 1 and body["strategy"] == "hybrid"
    assert body["cached"] is False and body["latency_ms"] >= 0
    assert body["recommendations"][0]["explanation"]


def test_recommend_default_limit_and_strategy(client):
    body = client.get("/recommend/4?strategy=skill_based").get_json()
    assert body["count"] == 5 and body["strategy"] == "skill_based"


def test_recommend_second_call_is_cached(client):
    client.get("/recommend/2")
    assert client.get("/recommend/2").get_json()["cached"] is True


def test_recommend_unknown_user_404(client):
    response = client.get("/recommend/9999")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "user_not_found"


def test_recommend_bad_requests_400(client):
    for url in (
        "/recommend/abc",
        "/recommend/1?limit=0",
        "/recommend/1?limit=999",
        "/recommend/1?limit=x",
        "/recommend/1?strategy=magic",
    ):
        response = client.get(url)
        assert response.status_code == 400, url
        assert response.get_json()["error"]["code"] == "invalid_request"


def test_feedback_created_and_changes_recommendations(client):
    top = client.get("/recommend/14?limit=1").get_json()["recommendations"][0]
    response = _post(
        client,
        {"user_id": 14, "content_id": top["content_id"], "type": "complete"},
    )
    body = response.get_json()
    assert response.status_code == 201 and body["status"] == "recorded"
    assert body["interaction"]["content_id"] == top["content_id"]
    after = client.get("/recommend/14?limit=5").get_json()
    assert after["cached"] is False and after["strategy"] == "hybrid"
    assert top["content_id"] not in [r["content_id"] for r in after["recommendations"]]


def test_feedback_with_rating(client):
    response = _post(
        client, {"user_id": 1, "content_id": 5, "type": "rate", "rating": 4.5}
    )
    assert response.status_code == 201
    assert response.get_json()["interaction"]["rating"] == 4.5


def test_feedback_bad_requests_400(client):
    bad = [
        {},
        {"user_id": 1},
        {"user_id": "1", "content_id": 1, "type": "like"},
        {"user_id": 1, "content_id": True, "type": "like"},
        {"user_id": 1, "content_id": 1, "type": 5},
        {"user_id": 1, "content_id": 1, "type": "share"},
        {"user_id": 1, "content_id": 1, "type": "rate"},
        {"user_id": 1, "content_id": 1, "type": "rate", "rating": 11},
    ]
    for payload in bad:
        assert _post(client, payload).status_code == 400, payload
    assert client.post("/feedback", data="not json").status_code == 400
    assert client.post("/feedback", json=[1, 2]).status_code == 400


def test_feedback_unknown_user_or_content_404(client):
    assert (
        _post(client, {"user_id": 999, "content_id": 1, "type": "like"}).status_code
        == 404
    )
    assert (
        _post(client, {"user_id": 1, "content_id": 999, "type": "like"}).status_code
        == 404
    )


def test_request_id_generated_and_propagated(client):
    generated = client.get("/health")
    assert len(generated.headers["X-Request-ID"]) == 12
    assert generated.get_json()["request_id"] == generated.headers["X-Request-ID"]
    custom = client.get("/recommend/1", headers={"X-Request-ID": "trace-42"})
    assert custom.headers["X-Request-ID"] == "trace-42"
    assert custom.get_json()["request_id"] == "trace-42"
    error = client.get("/recommend/9999", headers={"X-Request-ID": "e1"})
    assert error.get_json()["error"]["request_id"] == "e1"


def test_unknown_route_and_method_return_json(client):
    missing = client.get("/nope")
    assert missing.status_code == 404 and "error" in missing.get_json()
    wrong = client.post("/health")
    assert wrong.status_code == 405 and "error" in wrong.get_json()


def test_metrics_endpoint(client):
    client.get("/recommend/1")
    client.get("/recommend/1")
    client.get("/recommend/9999")
    body = client.get("/metrics").get_json()
    endpoint = body["endpoints"]["/recommend/<user_id>"]
    assert endpoint["count"] == 3 and endpoint["p95_ms"] >= endpoint["p50_ms"]
    assert body["status_codes"]["404"] == 1
    assert body["recommender"]["cache"]["hits"] == 1
    assert body["total_requests"] >= 3


def test_api_key_authentication(orchestrator):
    app = create_app(orchestrator=orchestrator, api_key="secret")
    client = app.test_client()
    assert client.get("/health").status_code == 200  # health stays open
    assert client.get("/recommend/1").status_code == 401
    assert client.get("/metrics").status_code == 401
    assert client.post("/feedback", json={}).status_code == 401
    wrong = client.get("/recommend/1", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    ok = client.get("/recommend/1", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_unhandled_error_returns_500_json(orchestrator, monkeypatch):
    app = create_app(orchestrator=orchestrator, api_key="")
    monkeypatch.setattr(
        orchestrator,
        "get_recommendations_detailed",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    response = app.test_client().get("/recommend/1")
    assert response.status_code == 500
    assert response.get_json()["error"]["code"] == "internal_error"


def test_create_app_builds_its_own_database(tmp_path):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    app = create_app(database_url=url, api_key="", cache_ttl=5)
    response = app.test_client().get("/health")
    assert response.status_code == 200 and response.get_json()["users"] == 0
