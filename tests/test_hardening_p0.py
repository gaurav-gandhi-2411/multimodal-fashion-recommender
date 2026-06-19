"""tests/test_hardening_p0.py -- Serve-path HTTP tests for P0 hardening.

Tests (all exercise the live HTTP handler via TestClient):
  H1: k bounds (ge=1, le=100) on /similar and /visual-search -> 422 outside range
  H2: image upload >10 MB -> 413
  H3: FAISS search exception -> 503 (not 500)
  H5: /health strips brand inventory (no 'brands' key)
  C2: rate limiter returns 429 after limit exceeded (isolated test app)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rerank import RerankConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(200, 100, 50)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _oversized_bytes(size_mb: int = 11) -> bytes:
    """Return a raw byte string > size_mb MB (not a valid image -- just for size check)."""
    return b"X" * (size_mb * 1024 * 1024)


def _fixed_vec(dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_vs_state(known_ids: list[int]) -> MagicMock:
    """Minimal BrandState mock for visual-search tests."""
    state = MagicMock()
    state.api_key = "p0-test-key"
    state.config.brand = "testbrand"
    state.art_map = {
        aid: {"title": f"Item {aid}", "category": "Shirts", "price_inr": 999.0}
        for aid in known_ids
    }
    state.config.rerank = RerankConfig(enabled=False)
    state.visual_retriever = MagicMock()
    state.visual_retriever.search.return_value = [
        (aid, 0.9 - 0.01 * i) for i, aid in enumerate(known_ids)
    ]
    return state


def _make_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


# ---------------------------------------------------------------------------
# H1: k bounds on /similar
# ---------------------------------------------------------------------------

def test_similar_k_zero_returns_422(api_client) -> None:
    resp = api_client.get(
        "/v1/test_brand/item/111/similar?k=0",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 422, f"Expected 422 for k=0, got {resp.status_code}"


def test_similar_k_101_returns_422(api_client) -> None:
    resp = api_client.get(
        "/v1/test_brand/item/111/similar?k=101",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 422, f"Expected 422 for k=101, got {resp.status_code}"


def test_similar_k_1_is_valid(api_client) -> None:
    resp = api_client.get(
        "/v1/test_brand/item/111/similar?k=1",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 200, f"Expected 200 for k=1, got {resp.status_code}"


def test_similar_k_100_is_valid(api_client) -> None:
    resp = api_client.get(
        "/v1/test_brand/item/111/similar?k=100",
        headers={"X-Api-Key": "test-api-key-123"},
    )
    assert resp.status_code == 200, f"Expected 200 for k=100, got {resp.status_code}"


# ---------------------------------------------------------------------------
# H1: k bounds on /visual-search
# ---------------------------------------------------------------------------

def test_visual_search_k_zero_returns_422() -> None:
    state = _make_vs_state([1, 2, 3])
    registry = _make_registry(state)
    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_fixed_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=0",
                files={"image": ("t.png", _tiny_png_bytes(), "image/png")},
                headers={"X-Api-Key": "p0-test-key"},
            )
    assert resp.status_code == 422, f"Expected 422 for k=0, got {resp.status_code}"


def test_visual_search_k_101_returns_422() -> None:
    state = _make_vs_state([1, 2, 3])
    registry = _make_registry(state)
    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_fixed_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=101",
                files={"image": ("t.png", _tiny_png_bytes(), "image/png")},
                headers={"X-Api-Key": "p0-test-key"},
            )
    assert resp.status_code == 422, f"Expected 422 for k=101, got {resp.status_code}"


# ---------------------------------------------------------------------------
# H2: image size cap
# ---------------------------------------------------------------------------

def test_visual_search_oversized_image_returns_413() -> None:
    state = _make_vs_state([1])
    registry = _make_registry(state)
    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("big.bin", _oversized_bytes(11), "application/octet-stream")},
                headers={"X-Api-Key": "p0-test-key"},
            )
    assert resp.status_code == 413, f"Expected 413 for oversized upload, got {resp.status_code}"
    assert "large" in resp.json()["detail"].lower()


def test_visual_search_10mb_exactly_is_accepted() -> None:
    """10 MB exactly should NOT be rejected (limit is strictly > 10 MB)."""
    state = _make_vs_state([1])
    registry = _make_registry(state)
    exactly_10mb = b"X" * (10 * 1024 * 1024)
    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("ten.bin", exactly_10mb, "application/octet-stream")},
                headers={"X-Api-Key": "p0-test-key"},
            )
    # 10 MB exactly: not rejected for size. Will get 400 (invalid image) because it's not a real image.
    assert resp.status_code == 400, f"Expected 400 (invalid image), got {resp.status_code}"


# ---------------------------------------------------------------------------
# H3: FAISS error handling
# ---------------------------------------------------------------------------

def _make_similar_state(known_ids: list[int]) -> MagicMock:
    """Minimal BrandState mock for /similar tests."""
    state = MagicMock()
    state.api_key = "p0-test-key"
    state.config.brand = "testbrand"
    state.art_map = {
        aid: {"title": f"Item {aid}", "category": "Shirts", "price_inr": 999.0}
        for aid in known_ids
    }
    from app.rerank import RerankConfig
    state.config.rerank = RerankConfig(enabled=False)
    state.retriever = MagicMock()
    state.retriever.search.return_value = [
        (aid, 0.9 - 0.01 * i) for i, aid in enumerate(known_ids)
    ]
    state.retriever.index.reconstruct.side_effect = lambda row: np.zeros(256, dtype=np.float32)
    state.faiss_aid_to_row = {aid: i for i, aid in enumerate(known_ids)}
    state.user_history = None
    return state


def test_similar_faiss_error_returns_503() -> None:
    """When FAISS .search() raises, the route must return 503, not 500."""
    known_ids = [1, 2, 3]
    state = _make_similar_state(known_ids)
    state.retriever.search.side_effect = RuntimeError("FAISS index corrupted")
    registry = _make_registry(state)
    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/v1/{state.config.brand}/item/1/similar",
                headers={"X-Api-Key": "p0-test-key"},
            )
    assert resp.status_code == 503, (
        f"Expected 503 for FAISS failure, got {resp.status_code}: {resp.text}"
    )
    assert "unavailable" in resp.json()["detail"].lower()


def test_visual_search_faiss_error_returns_503() -> None:
    """When visual_retriever.search() raises, the route must return 503."""
    state = _make_vs_state([1])
    state.visual_retriever.search.side_effect = RuntimeError("FAISS OOM")
    registry = _make_registry(state)
    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_fixed_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("q.png", _tiny_png_bytes(), "image/png")},
                headers={"X-Api-Key": "p0-test-key"},
            )
    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# H5: /health strips brand inventory
# ---------------------------------------------------------------------------

def test_health_has_no_brand_inventory(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "brands" not in data, "Health endpoint must not expose brand inventory unauthenticated"


# ---------------------------------------------------------------------------
# C2: rate limiter -- isolated test app with 1/minute limit
# ---------------------------------------------------------------------------

def test_rate_limit_returns_429_after_limit_exceeded() -> None:
    """A 1/minute limit must return 429 on the second request from the same IP."""
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    test_limiter = Limiter(key_func=get_remote_address)
    probe_app = FastAPI()
    probe_app.state.limiter = test_limiter
    probe_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    probe_app.add_middleware(SlowAPIMiddleware)

    @probe_app.get("/probe")
    @test_limiter.limit("1/minute")
    async def probe(request: Request):
        return {"ok": True}

    with TestClient(probe_app, raise_server_exceptions=False) as client:
        r1 = client.get("/probe")
        r2 = client.get("/probe")

    assert r1.status_code == 200, f"First request should succeed, got {r1.status_code}"
    assert r2.status_code == 429, f"Second request must be rate-limited, got {r2.status_code}"
