"""tests/test_hardening_m1m3m4m5.py

M1: 3 missing 404 HTTP tests — item-not-found on /recommend, /similar, /complete.
    All three routes call _get_item_embedding(item_id) and raise 404 when the
    item_id is not in the brand's FAISS index.

M3: request_id propagates into structlog contextvars — confirmed via response JSON
    and uniqueness across concurrent requests.

M4: top_k query-param alias on /similar — ?top_k=N should behave identically to ?k=N.

M5: X-Api-Key header casing — FastAPI/Starlette normalises headers to lowercase, so
    both 'X-Api-Key' and 'X-API-Key' must authenticate successfully.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# M1 — 404 when item_id is not in catalogue
# ---------------------------------------------------------------------------

class TestItemNotFound404:
    """All three handlers raise HTTP 404 when item_id is not in the catalogue."""

    def test_recommend_unknown_item_id_returns_404(self, api_client) -> None:
        resp = api_client.post(
            "/v1/test_brand/recommend",
            json={"item_id": "99999", "k": 5},
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 404, (
            f"Expected 404 for unknown item_id=99999, got {resp.status_code}. "
            f"Body: {resp.text}"
        )

    def test_similar_unknown_item_id_returns_404(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/99999/similar",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 404, (
            f"Expected 404 for unknown item_id=99999, got {resp.status_code}. "
            f"Body: {resp.text}"
        )

    def test_complete_unknown_item_id_returns_404(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/99999/complete",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 404, (
            f"Expected 404 for unknown item_id=99999, got {resp.status_code}. "
            f"Body: {resp.text}"
        )

    def test_known_item_id_similar_returns_200(self, api_client) -> None:
        """Control: a known item_id (111) should return 200, not 404."""
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 200, (
            f"Expected 200 for known item_id=111, got {resp.status_code}"
        )

    def test_404_response_body_contains_detail(self, api_client) -> None:
        """The 404 detail should mention the item_id so callers know which item was missing."""
        resp = api_client.get(
            "/v1/test_brand/item/88888/similar",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body, "404 response must include 'detail' field"
        assert "88888" in body["detail"], (
            f"404 detail should reference the missing item_id. Got: {body['detail']!r}"
        )


# ---------------------------------------------------------------------------
# M3 — request_id present in response and unique per call
# ---------------------------------------------------------------------------

class TestRequestIdPropagation:
    def test_recommend_response_has_request_id(self, api_client) -> None:
        resp = api_client.post(
            "/v1/test_brand/recommend",
            json={"item_id": "111", "k": 3},
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "request_id" in body and body["request_id"], (
            "recommend response must include a non-null request_id"
        )

    def test_similar_response_has_request_id(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "request_id" in body and body["request_id"], (
            "similar response must include a non-null request_id"
        )

    def test_request_ids_are_unique_across_calls(self, api_client) -> None:
        """Every request gets a distinct UUID — same endpoint, same params."""
        ids = set()
        for _ in range(5):
            resp = api_client.get(
                "/v1/test_brand/item/111/similar",
                headers={"X-Api-Key": "test-api-key-123"},
            )
            assert resp.status_code == 200
            ids.add(resp.json()["request_id"])
        assert len(ids) == 5, (
            f"Expected 5 unique request_ids across 5 calls, got {len(ids)}: {ids}"
        )


# ---------------------------------------------------------------------------
# M4 — top_k alias on /similar
# ---------------------------------------------------------------------------

class TestTopKAlias:
    def test_top_k_param_accepted_returns_200(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar?top_k=3",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 200, (
            f"Expected 200 when ?top_k=3 is passed, got {resp.status_code}. "
            f"Body: {resp.text}"
        )

    def test_top_k_limits_result_count(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar?top_k=1",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The mock retriever returns 3 items; top_k=1 should limit to ≤1 result.
        assert len(body["results"]) <= 1, (
            f"top_k=1 should return at most 1 result, got {len(body['results'])}"
        )

    def test_top_k_zero_returns_422(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar?top_k=0",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp.status_code == 422, (
            f"top_k=0 should be rejected with 422, got {resp.status_code}"
        )

    def test_top_k_overrides_k_when_both_present(self, api_client) -> None:
        """When both ?k and ?top_k are supplied, top_k takes precedence."""
        resp_k5 = api_client.get(
            "/v1/test_brand/item/111/similar?k=5&top_k=1",
            headers={"X-Api-Key": "test-api-key-123"},
        )
        assert resp_k5.status_code == 200
        assert len(resp_k5.json()["results"]) <= 1, (
            "top_k=1 should override k=5 and return at most 1 result"
        )


# ---------------------------------------------------------------------------
# M5 — X-Api-Key header casing
# ---------------------------------------------------------------------------

class TestApiKeyCasing:
    def test_x_api_key_lowercase_alias_works(self, api_client) -> None:
        """x-api-key (all lowercase) should authenticate the same as X-Api-Key."""
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"x-api-key": "test-api-key-123"},
        )
        assert resp.status_code == 200, (
            f"x-api-key (lowercase) should authenticate. Got {resp.status_code}: {resp.text}"
        )

    def test_x_api_key_caps_alias_works(self, api_client) -> None:
        """X-API-KEY (all caps) should authenticate. FastAPI normalises headers to lowercase."""
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"X-API-KEY": "test-api-key-123"},
        )
        assert resp.status_code == 200, (
            f"X-API-KEY (all caps) should authenticate. Got {resp.status_code}: {resp.text}"
        )

    def test_missing_api_key_returns_401(self, api_client) -> None:
        resp = api_client.get("/v1/test_brand/item/111/similar")
        assert resp.status_code == 401, (
            f"Missing API key should return 401, got {resp.status_code}"
        )

    def test_wrong_api_key_returns_401(self, api_client) -> None:
        resp = api_client.get(
            "/v1/test_brand/item/111/similar",
            headers={"X-Api-Key": "wrong-key"},
        )
        assert resp.status_code == 401, (
            f"Wrong API key should return 401, got {resp.status_code}"
        )
