"""tests/test_style_search.py -- Serve-path tests for /v1/{brand}/style-search.

Tests the CLIP text-query → visual-FAISS retrieval path including:
  - happy-path response shape and field presence
  - match_confidence score-gap semantics (same formula as C3 visual-search)
  - HTTP 503 when visual_retriever is unconfigured
  - HTTP 422 for empty or oversized text
  - color rerank applied when ?color= param is passed
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rerank import RerankConfig  # noqa: E402


def _vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_state(scores: list[float], color_index: dict | None = None) -> MagicMock:
    ids = list(range(1, len(scores) + 1))
    state = MagicMock()
    state.api_key = "style-test-key"
    state.config.brand = "stylebrand"
    state.art_map = {
        aid: {"title": f"Item {aid}", "category": "Shirts", "price_inr": 999.0}
        for aid in ids
    }
    state.config.rerank = RerankConfig(enabled=False)
    state.visual_retriever = MagicMock()
    state.visual_retriever.search.return_value = list(zip(ids, scores))
    state.color_index = color_index or {}
    return state


def _make_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


def _call_style_search(
    state: MagicMock,
    text: str = "white oversized cotton shirt",
    k: int = 5,
    color: str | None = None,
) -> tuple[int, dict]:
    registry = _make_registry(state)
    url = f"/v1/{state.config.brand}/style-search?text={text}&k={k}"
    if color:
        url += f"&color={color}"
    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_text", return_value=_vec()),
    ):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(url, headers={"X-Api-Key": "style-test-key"})
    return resp.status_code, resp.json()


def test_style_search_returns_200() -> None:
    state = _make_state([0.88, 0.82, 0.78, 0.74, 0.70])
    code, body = _call_style_search(state)
    assert code == 200, body


def test_style_search_response_shape() -> None:
    state = _make_state([0.88, 0.82, 0.78, 0.74, 0.70])
    _, body = _call_style_search(state)
    assert "results" in body
    assert "match_confidence" in body
    assert "query" in body
    assert body["query"] == "white oversized cotton shirt"
    assert body["brand"] == "stylebrand"
    assert isinstance(body["results"], list)
    assert len(body["results"]) == 5


def test_style_search_match_confidence_present() -> None:
    state = _make_state([0.90, 0.85, 0.80, 0.75, 0.70])
    _, body = _call_style_search(state)
    assert "match_confidence" in body
    assert isinstance(body["match_confidence"], float)


def test_style_search_match_confidence_is_score_gap() -> None:
    scores = [0.90, 0.85, 0.80, 0.75, 0.70]
    state = _make_state(scores)
    _, body = _call_style_search(state, k=5)
    expected = round(0.90 - 0.70, 4)
    assert abs(body["match_confidence"] - expected) < 0.001, (
        f"Expected gap {expected}, got {body['match_confidence']}"
    )


def test_style_search_high_confidence_for_clear_match() -> None:
    scores = [0.97, 0.72, 0.70, 0.69, 0.68]
    state = _make_state(scores)
    _, body = _call_style_search(state, k=5)
    assert body["match_confidence"] > 0.20


def test_style_search_low_confidence_for_tight_cluster() -> None:
    # Simulates "your catalog lacks items in this style" — scores are tightly bunched
    scores = [0.72, 0.715, 0.71, 0.705, 0.70]
    state = _make_state(scores)
    _, body = _call_style_search(state, k=5)
    assert body["match_confidence"] < 0.05


def test_style_search_503_when_no_visual_retriever() -> None:
    state = _make_state([0.88, 0.82])
    state.visual_retriever = None
    code, body = _call_style_search(state)
    assert code == 503, body


def test_style_search_empty_text_rejected() -> None:
    state = _make_state([0.88, 0.82])
    code, _ = _call_style_search(state, text="")
    assert code == 422


def test_style_search_color_rerank_called_when_color_param_given() -> None:
    state = _make_state([0.88, 0.82, 0.78, 0.74, 0.70])
    # Provide a dummy color index so the branch fires
    state.color_index = {1: (0.5, 0.8, 0.9), 2: (0.1, 0.6, 0.8)}
    with (
        patch("app.api.main.load_registry", return_value=_make_registry(state)),
        patch("app.visual.encode_query_text", return_value=_vec()),
        patch("app.api.routes.color_rerank", return_value=[(1, 0.9), (2, 0.8)]) as mock_cr,
    ):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/style-search?text=red+shirt&k=5&color=ff0000",
                headers={"X-Api-Key": "style-test-key"},
            )
    assert resp.status_code == 200
    mock_cr.assert_called_once()
