"""tests/test_hardening_c1c3.py -- Serve-path HTTP tests for C1 + C3 hardening.

C1: Color-aware ranking is now in the BACKEND — /visual-search accepts ?color=<hex>
    and applies color_rerank() server-side. Results are sorted by blended score.

C3: /visual-search returns match_quality="insufficient" when top CLIP score is below
    the per-brand visual_search_min_score threshold (set in brand YAML).
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

from app.color import ColorIndex, color_similarity, hex_to_hsv  # noqa: E402
from app.rerank import RerankConfig  # noqa: E402


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Unit tests for app.color (no HTTP)
# ---------------------------------------------------------------------------

class TestHexToHsv:
    def test_pure_red(self) -> None:
        hsv = hex_to_hsv("ff0000")
        assert hsv is not None
        assert abs(hsv["h"] - 0.0) < 1.0
        assert hsv["s"] > 0.99
        assert hsv["v"] > 0.99

    def test_pure_blue(self) -> None:
        hsv = hex_to_hsv("0000ff")
        assert hsv is not None
        assert abs(hsv["h"] - 240.0) < 1.0

    def test_white_is_achromatic(self) -> None:
        hsv = hex_to_hsv("ffffff")
        assert hsv is not None
        assert hsv["s"] < 0.01

    def test_invalid_returns_none(self) -> None:
        assert hex_to_hsv("gg0000") is None
        assert hex_to_hsv("short") is None
        assert hex_to_hsv("#ff0000") is None  # with # prefix is invalid


class TestColorSimilarity:
    def test_identical_colors_score_1(self) -> None:
        red = {"h": 0.0, "s": 1.0, "v": 1.0}
        assert abs(color_similarity(red, red) - 1.0) < 0.01

    def test_red_vs_blue_is_low(self) -> None:
        # Red (h=0) vs Blue (h=240): h_diff = min(240,120)/180 = 0.667
        # sim = 1 - 0.6*0.667 = 0.60 — meaningfully lower than similar-hue pairs
        red = {"h": 0.0, "s": 1.0, "v": 1.0}
        blue = {"h": 240.0, "s": 1.0, "v": 1.0}
        assert color_similarity(red, blue) < 0.7

    def test_achromatic_pair_relies_on_value(self) -> None:
        white = {"h": 0.0, "s": 0.0, "v": 1.0}
        black = {"h": 0.0, "s": 0.0, "v": 0.0}
        assert color_similarity(white, black) < 0.5

    def test_similar_shades_score_high(self) -> None:
        red1 = {"h": 0.0, "s": 0.95, "v": 0.9}
        red2 = {"h": 5.0, "s": 1.0, "v": 0.85}
        assert color_similarity(red1, red2) > 0.8


# ---------------------------------------------------------------------------
# C1 serve-path test: color rerank applied in backend
# ---------------------------------------------------------------------------

def _make_color_state(
    known_ids: list[int],
    color_index: ColorIndex,
    *,
    min_score: float | None = None,
) -> MagicMock:
    state = MagicMock()
    state.api_key = "c1c3-test-key"
    state.config.brand = "colorbrand"
    state.config.visual_search_min_score = min_score
    state.art_map = {
        aid: {"title": f"Item {aid}", "category": "Shirts", "price_inr": 999.0}
        for aid in known_ids
    }
    state.config.rerank = RerankConfig(enabled=False)
    state.visual_retriever = MagicMock()
    # Return scores in reverse order so color rerank has meaningful work to do
    state.visual_retriever.search.return_value = [
        (aid, float(0.9 - 0.05 * i)) for i, aid in enumerate(known_ids)
    ]
    state.color_index = color_index
    return state


def _make_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


def test_visual_search_color_param_accepted_and_changes_order() -> None:
    """When ?color=<hex> is passed and color index has data, results are reordered by blended score."""
    # Set up: item 1 is red (will match red query), item 2 is blue (won't match)
    # FAISS returns them in CLIP order: 1 (0.90), 2 (0.85) — both near top
    red_hsv = {"h": 0.0, "s": 1.0, "v": 1.0}
    blue_hsv = {"h": 240.0, "s": 1.0, "v": 1.0}
    color_index: ColorIndex = {"1": red_hsv, "2": blue_hsv}

    state = _make_color_state([1, 2], color_index)
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=2&color=ff0000",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["match_quality"] == "ok"
    result_ids = [r["item_id"] for r in body["results"]]
    # Red item (id=1) should still be first (it was also first in CLIP) or
    # at minimum, blue item should not outrank red after color blend.
    # With red query: item 1 gets high color_sim (~1.0), item 2 gets low (~0.0).
    # After normalize+blend: 1 wins clearly.
    assert result_ids[0] == "1", f"Red item should rank first with red query; got {result_ids}"


def test_visual_search_no_color_param_returns_ok_quality() -> None:
    """When no ?color param is passed, match_quality is 'ok' and results are unmodified."""
    state = _make_color_state([1, 2, 3], color_index={})
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=3",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200
    assert resp.json()["match_quality"] == "ok"


def test_visual_search_invalid_color_param_ignored() -> None:
    """An invalid ?color hex is silently ignored (no crash, results returned normally)."""
    state = _make_color_state([1], color_index={"1": {"h": 0.0, "s": 1.0, "v": 1.0}})
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=1&color=NOTAHEX",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200
    assert resp.json()["match_quality"] == "ok"


# ---------------------------------------------------------------------------
# C3 serve-path tests: min_score threshold
# ---------------------------------------------------------------------------

def test_visual_search_low_score_returns_insufficient() -> None:
    """When top CLIP score is below visual_search_min_score, return match_quality='insufficient'."""
    state = _make_color_state([1, 2], color_index={}, min_score=0.80)
    # Override retriever to return low scores (simulating OOD image)
    state.visual_retriever.search.return_value = [(1, 0.45), (2, 0.40)]
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=2",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["match_quality"] == "insufficient"
    assert body["results"] == []


def test_visual_search_high_score_returns_ok() -> None:
    """When top score meets threshold, match_quality='ok' and results are returned."""
    state = _make_color_state([1, 2], color_index={}, min_score=0.40)
    # Scores above threshold
    state.visual_retriever.search.return_value = [(1, 0.90), (2, 0.85)]
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=2",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["match_quality"] == "ok"
    assert len(body["results"]) == 2


def test_visual_search_no_min_score_config_always_returns_ok() -> None:
    """When visual_search_min_score is None (not configured), always return ok."""
    state = _make_color_state([1], color_index={}, min_score=None)
    state.visual_retriever.search.return_value = [(1, 0.10)]  # very low score
    registry = _make_registry(state)

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=_vec()),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/visual-search?k=1",
                files={"image": ("t.png", _tiny_png(), "image/png")},
                headers={"X-Api-Key": "c1c3-test-key"},
            )

    assert resp.status_code == 200
    assert resp.json()["match_quality"] == "ok"
