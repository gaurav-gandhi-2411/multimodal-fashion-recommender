"""tests/test_visual.py -- Route-level + unit tests for /v1/{brand}/visual-search.

Phase-7 pure-CLIP guardrails:
  - Route tests exercise the feature through the REAL HTTP route (TestClient),
    with encode_query_image MOCKED so CI never needs GPU/large model weights.
  - The 503 test verifies state.visual_retriever is None -> HTTP 503 (not a 500).
  - A separate optional test (guarded by pytest.importorskip) exercises the real
    CLIP encoder and checks the output is (512,) unit-norm float32 -- no tower/SBERT.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.rerank import RerankConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_png_bytes() -> bytes:
    """Create a minimal in-memory 16x16 RGB PNG that any image decoder accepts."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _fixed_query_vector(dim: int = 512) -> np.ndarray:
    """Return a deterministic L2-normalised vector (same dim as the CLIP-512 FAISS index)."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_state(known_ids: list[int], *, has_visual_retriever: bool = True) -> MagicMock:
    """Build a minimal MagicMock BrandState for visual-search route tests.

    Args:
        known_ids: article IDs the mock retriever will return.
        has_visual_retriever: when False, state.visual_retriever is set to None
            so the route returns HTTP 503.
    """
    art_map = {
        aid: {
            "title": f"Item {aid}",
            "category": "Shirts",
            "price_inr": 999.0,
            "pdp_url": f"https://example.com/{aid}",
        }
        for aid in known_ids
    }
    state = MagicMock()
    state.api_key = "vs-test-key"
    state.config.brand = "testbrand"
    state.art_map = art_map
    # Disable reranking so these route-plumbing tests are not affected by
    # the inferred-category path added in the pure-image visual-search fix.
    state.config.rerank = RerankConfig(enabled=False)
    # C1: empty color index so color_rerank is a no-op in these plumbing tests.
    state.color_index = {}

    if has_visual_retriever:
        state.visual_retriever = MagicMock()
        state.visual_retriever.search.return_value = [
            (aid, float(1.0 - 0.01 * i)) for i, aid in enumerate(known_ids)
        ]
    else:
        # Simulate brand with no visual index built yet.
        state.visual_retriever = None

    return state


def _make_registry(state: MagicMock) -> MagicMock:
    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == state.config.brand else None
    registry.brand_names.return_value = [state.config.brand]
    return registry


# ---------------------------------------------------------------------------
# Route tests (encoders fully mocked; exercises the live HTTP handler)
# ---------------------------------------------------------------------------


def test_visual_search_route_200_with_expected_item() -> None:
    """POST /v1/{brand}/visual-search returns 200 and expected item ids.

    encode_query_image is patched to return a fixed 512-d CLIP vector so this
    test never loads CLIP -- fast in CI, covers the full HTTP path.
    """
    known_ids = [101, 202, 303]
    state = _make_state(known_ids)
    registry = _make_registry(state)
    fixed_vec = _fixed_query_vector()

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=fixed_vec),
    ):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("test.png", _tiny_png_bytes(), "image/png")},
                params={"k": 3},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["brand"] == state.config.brand
    assert "request_id" in body
    assert "latency_ms" in body
    returned_ids = [item["item_id"] for item in body["results"]]
    # All expected ids should appear (retriever mock returns them all)
    assert "101" in returned_ids, f"Expected item '101' in results: {returned_ids}"
    assert len(body["results"]) == len(known_ids)
    # Confirm pdp_url is forwarded from art_map
    result_101 = next(r for r in body["results"] if r["item_id"] == "101")
    assert result_101["pdp_url"] == "https://example.com/101"


def test_visual_search_route_calls_visual_retriever_with_fixed_vector() -> None:
    """encode_query_image's return value is forwarded verbatim to visual_retriever.search."""
    known_ids = [42]
    state = _make_state(known_ids)
    registry = _make_registry(state)
    fixed_vec = _fixed_query_vector()

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=fixed_vec) as mock_enc,
    ):
        from app.api.main import app

        with TestClient(app) as client:
            client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("q.png", _tiny_png_bytes(), "image/png")},
                headers={"X-Api-Key": "vs-test-key"},
            )

    # encode_query_image must have been called once
    mock_enc.assert_called_once()
    # visual_retriever.search must have received the fixed vector
    call_args = state.visual_retriever.search.call_args
    np.testing.assert_array_equal(call_args[0][0], fixed_vec)


def test_visual_search_route_503_when_no_visual_retriever() -> None:
    """POST returns HTTP 503 when state.visual_retriever is None (index not built)."""
    known_ids = [1]
    state = _make_state(known_ids, has_visual_retriever=False)
    registry = _make_registry(state)
    fixed_vec = _fixed_query_vector()

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=fixed_vec),
    ):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("q.png", _tiny_png_bytes(), "image/png")},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 503, resp.text
    assert "visual search not configured" in resp.json()["detail"].lower()


def test_visual_search_route_400_on_non_image_bytes() -> None:
    """Posting garbage bytes (not a valid image) returns HTTP 400."""
    known_ids = [1]
    state = _make_state(known_ids)
    registry = _make_registry(state)

    # Do NOT mock encode_query_image -- we want the real ValueError to propagate to 400.
    # But encode_query_image lazily imports open_clip; to avoid that dep in unit tests
    # we post bytes that PIL will reject, triggering the ValueError before any import.
    garbage = b"this is not an image"

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("bad.bin", garbage, "application/octet-stream")},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 400, resp.text


def test_visual_search_route_401_without_api_key() -> None:
    """Missing X-Api-Key header returns 401."""
    known_ids = [1]
    state = _make_state(known_ids)
    registry = _make_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search",
                files={"image": ("q.png", _tiny_png_bytes(), "image/png")},
            )

    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# Optional real-encoder unit test (guarded; skipped when open_clip is absent)
# ---------------------------------------------------------------------------


def test_encode_query_image_shape_and_unit_norm() -> None:
    """encode_query_image returns a (512,) float32 vector with L2-norm ~= 1.0.

    Pure CLIP-512 path: no item tower, no SBERT. Requires open_clip; skipped when absent.
    """
    pytest.importorskip("open_clip", reason="open_clip not installed")

    from app.visual import encode_query_image

    png_bytes = _tiny_png_bytes()
    vec = encode_query_image(png_bytes)

    assert vec.shape == (512,), f"Expected (512,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"
