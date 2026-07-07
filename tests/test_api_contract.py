from __future__ import annotations

import uuid


def test_recommend_200_with_item_id(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111", "k": 3},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["brand"] == "test_brand"
    assert data["cold_start"] is True
    assert isinstance(data["results"], list)


def test_recommend_response_has_request_id(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111", "k": 3},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    data = resp.json()
    assert "request_id" in data
    uuid.UUID(data["request_id"])  # raises ValueError if not a valid UUID


def test_recommend_response_schema(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111", "k": 3},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    data = resp.json()
    assert "request_id" in data
    assert "brand" in data
    assert "results" in data
    assert "cold_start" in data
    assert "latency_ms" in data
    result = data["results"][0]
    assert "item_id" in result
    assert "score" in result


def test_recommend_result_item_ids_are_strings(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111", "k": 3},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    for item in resp.json()["results"]:
        assert isinstance(item["item_id"], str)


def test_recommend_includes_pdp_url(api_client, mock_art_map):
    """Regression test: /recommend silently dropped pdp_url for every result even
    though /visual-search, /style-search, and /complete all populate it from the
    same art_map (found in the 2026-07-07 audit)."""
    mock_art_map[222]["pdp_url"] = "https://example.com/products/red-shirt"
    try:
        resp = api_client.post(
            "/v1/test_brand/recommend",
            json={"item_id": "111", "k": 3},
            headers={"X-Api-Key": "test-api-key-123"},
        )
        results = resp.json()["results"]
        by_id = {r["item_id"]: r for r in results}
        assert by_id["222"]["pdp_url"] == "https://example.com/products/red-shirt"
    finally:
        del mock_art_map[222]["pdp_url"]


def test_recommend_422_when_no_user_or_item(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"k": 5},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 422


def test_similar_200(api_client):
    resp = api_client.get(
        "/v1/test_brand/item/222/similar?k=3",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_item_id"] == "222"
    assert "request_id" in data
    uuid.UUID(data["request_id"])  # must be a valid UUID


def test_similar_response_schema(api_client):
    resp = api_client.get(
        "/v1/test_brand/item/333/similar",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    data = resp.json()
    assert "request_id" in data
    assert "brand" in data
    assert "query_item_id" in data
    assert "results" in data
    assert "latency_ms" in data


def test_similar_excludes_query_item(api_client):
    resp = api_client.get(
        "/v1/test_brand/item/111/similar",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    result_ids = [r["item_id"] for r in resp.json()["results"]]
    assert "111" not in result_ids


def test_similar_includes_pdp_url(api_client, mock_art_map):
    """Regression test: /similar silently dropped pdp_url for every result even
    though /visual-search, /style-search, and /complete all populate it from the
    same art_map (found in the 2026-07-07 audit)."""
    mock_art_map[222]["pdp_url"] = "https://example.com/products/red-shirt"
    try:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        results = resp.json()["results"]
        by_id = {r["item_id"]: r for r in results}
        assert by_id["222"]["pdp_url"] == "https://example.com/products/red-shirt"
    finally:
        del mock_art_map[222]["pdp_url"]


def test_health_returns_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "brands" not in data  # brand inventory stripped from unauthenticated health
