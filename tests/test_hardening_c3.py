"""tests/test_hardening_c3.py -- Serve-path tests for C3 Option B: match_confidence signal.

C3 Option B design:
  match_confidence = top-1 CLIP score - min(top-k CLIP scores)
  - float field in VisualSearchResponse; always present (never gates results)
  - High gap => top result stands apart (likely a good match)
  - Near-zero gap => top-k scores are tightly clustered (CLIP uncertain)
  - CLIP always returns nearest fashion for any image; callers use this field
    as a UI signal, the API never withholds results based on it
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


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(200, 100, 50)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_state(scores: list[float]) -> MagicMock:
    """BrandState mock with visual_retriever returning the given scores for items 1..N."""
    ids = list(range(1, len(scores) + 1))
    state = MagicMock()
    state.api_key = "c3-test-key"
    state.config.brand = "c3brand"
    state.art_map = {
        aid: {"title": f"Item {aid}", "category": "Shirts", "price_inr": 999.0}
        for aid in ids
    }
    state.config.rerank = RerankConfig(enabled=False)
    state.visual_retriever = MagicMock()
    state.visual_retriever.search.return_value = list(zip(ids, scores))
    state.color_index = {}
    return state


def _make_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


def _call_visual_search(state: MagicMock, k: int = 5) -> dict:
    registry = _make_registry(state)
    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k={k}",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c3-test-key"},
            )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_match_confidence_present_in_response() -> None:
    """match_confidence field is always present in VisualSearchResponse."""
    state = _make_state([0.90, 0.85, 0.80, 0.75, 0.70])
    body = _call_visual_search(state, k=5)
    assert "match_confidence" in body


def test_match_confidence_is_score_gap() -> None:
    """match_confidence = top-1 score minus min(top-k scores)."""
    scores = [0.90, 0.85, 0.80, 0.75, 0.70]
    state = _make_state(scores)
    body = _call_visual_search(state, k=5)
    expected = round(0.90 - 0.70, 4)  # top-1 minus min-top-5
    assert abs(body["match_confidence"] - expected) < 0.001, (
        f"Expected gap {expected}, got {body['match_confidence']}"
    )


def test_match_confidence_high_for_clear_winner() -> None:
    """A single dominant score yields a high match_confidence."""
    # Top-1 much higher than the rest (like a catalogue self-match)
    scores = [0.97, 0.72, 0.70, 0.69, 0.68]
    state = _make_state(scores)
    body = _call_visual_search(state, k=5)
    # gap = 0.97 - 0.68 = 0.29; clearly high
    assert body["match_confidence"] > 0.20, (
        f"Expected high confidence for clear winner, got {body['match_confidence']}"
    )


def test_match_confidence_low_for_tight_cluster() -> None:
    """Tightly clustered scores yield low match_confidence (near-zero gap)."""
    # Simulates an OOD image: CLIP scores are all similar (no dominant match)
    scores = [0.72, 0.715, 0.71, 0.705, 0.70]
    state = _make_state(scores)
    body = _call_visual_search(state, k=5)
    # gap ≈ 0.72 - 0.70 = 0.02; low
    assert body["match_confidence"] < 0.05, (
        f"Expected low confidence for tight cluster, got {body['match_confidence']}"
    )


def test_results_always_returned_regardless_of_confidence() -> None:
    """API always returns results even when match_confidence is very low."""
    # Simulate worst-case OOD: all scores nearly identical
    scores = [0.650, 0.649, 0.648, 0.647, 0.646]
    state = _make_state(scores)
    body = _call_visual_search(state, k=5)
    assert len(body["results"]) == 5, "Results must always be returned"
    assert body["match_confidence"] < 0.01


def test_match_confidence_zero_for_single_result() -> None:
    """With only one result in pool, match_confidence is 0.0 (no gap to compute)."""
    state = _make_state([0.90])
    body = _call_visual_search(state, k=1)
    assert body["match_confidence"] == 0.0
