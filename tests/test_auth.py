from __future__ import annotations


def test_missing_api_key_returns_401(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111"},
        # no X-Api-Key header
    )
    assert resp.status_code == 401


def test_wrong_api_key_returns_401(api_client):
    resp = api_client.post(
        "/v1/test_brand/recommend",
        json={"item_id": "111"},
        headers={"X-Api-Key": "completely-wrong-key"},
    )
    assert resp.status_code == 401


def test_wrong_brand_returns_404(api_client):
    resp = api_client.post(
        "/v1/nonexistent_brand/recommend",
        json={"item_id": "111"},
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 404


def test_similar_missing_key_returns_401(api_client):
    resp = api_client.get("/v1/test_brand/item/111/similar")
    assert resp.status_code == 401


def test_similar_wrong_brand_returns_404(api_client):
    resp = api_client.get(
        "/v1/fake_brand/item/111/similar",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 404
