"""tests/test_attributes.py -- Unit tests for app/attributes.py.

classify_embeddings is tested with deterministic stub encoders (no FashionCLIP model
load) mirroring the fast/no-model-download convention used elsewhere in this test suite.
load_attribute_index is tested for the graceful-empty-on-missing-file behaviour that
mirrors app/color.py::load_color_index (see tests/test_hardening_c1c3.py for the analog).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import attributes as attrs_module
from app.attributes import (
    ATTRIBUTE_TAXONOMY,
    build_attribute_index,
    classify_embeddings,
    load_attribute_index,
)
from app.occasion import DEFAULT_OCCASION_LEXICON


class _OneHotTextEncoder:
    """Deterministic stand-in for FashionCLIPEncoder.encode_text: returns one-hot rows,
    one per prompt, in input order. No model load required."""

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)


class _RandomTextEncoder:
    """Deterministic (seeded) stand-in producing L2-normalised random text embeddings."""

    def __init__(self, dim: int, seed: int = 42) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        embs = self._rng.standard_normal((len(texts), self._dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return embs


def test_occasion_taxonomy_matches_canonical_set() -> None:
    """The 'occasion' category must reuse app/occasion.py's exact 5-label canonical set."""
    assert ATTRIBUTE_TAXONOMY["occasion"] == list(DEFAULT_OCCASION_LEXICON.keys())


def test_classify_embeddings_picks_matching_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a 3-label single-category taxonomy and one-hot text embeddings, an image
    embedding aligned with a given label must be classified as that label, and the
    confidence must equal top1_sim - top2_sim exactly."""
    small_taxonomy = {"shape": ["circle", "square", "triangle"]}
    small_templates = {"shape": "a {label} garment"}
    monkeypatch.setattr(attrs_module, "ATTRIBUTE_TAXONOMY", small_taxonomy)
    monkeypatch.setattr(attrs_module, "PROMPT_TEMPLATES", small_templates)

    # Item 0 aligns exactly with label 1 ("square") -> confidence 1.0 - 0.0.
    # Item 1 is mostly label 0 ("circle") with some label 1 mixed in -> confidence 0.9 - 0.1.
    image_embeddings = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    result = classify_embeddings(image_embeddings, _OneHotTextEncoder())

    assert set(result.keys()) == {"shape"}
    labels, confidences = result["shape"]
    assert labels[0] == "square"
    assert labels[1] == "circle"
    assert confidences[0] == pytest.approx(1.0, abs=1e-6)
    assert confidences[1] == pytest.approx(0.8, abs=1e-6)


def test_classify_embeddings_all_four_categories_present() -> None:
    """With the real ATTRIBUTE_TAXONOMY (4 categories), every category appears in the
    output with one (label, confidence) pair per item, labels drawn from the taxonomy,
    and non-negative confidences."""
    n_items = 3
    dim = 512
    rng = np.random.default_rng(42)
    image_embeddings = rng.standard_normal((n_items, dim)).astype(np.float32)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    result = classify_embeddings(image_embeddings, _RandomTextEncoder(dim))

    assert set(result.keys()) == set(ATTRIBUTE_TAXONOMY.keys())
    for category, labels in ATTRIBUTE_TAXONOMY.items():
        pred_labels, confidences = result[category]
        assert len(pred_labels) == n_items
        assert len(confidences) == n_items
        assert all(label in labels for label in pred_labels)
        assert all(c >= 0.0 for c in confidences)


def test_build_attribute_index_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_attribute_index assembles a JSON-keyed-by-item-id-string dict with the
    exact 8-key shape (4 labels + 4 confidences) per item."""
    small_taxonomy = {"shape": ["circle", "square"]}
    small_templates = {"shape": "a {label} garment"}
    monkeypatch.setattr(attrs_module, "ATTRIBUTE_TAXONOMY", small_taxonomy)
    monkeypatch.setattr(attrs_module, "PROMPT_TEMPLATES", small_templates)

    image_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    classified = classify_embeddings(image_embeddings, _OneHotTextEncoder())

    index = build_attribute_index([101, 202], classified)

    assert set(index.keys()) == {"101", "202"}
    assert index["101"] == {"shape": "circle", "shape_confidence": 1.0}
    assert index["202"] == {"shape": "square", "shape_confidence": 1.0}


def test_load_attribute_index_missing_file_returns_empty(tmp_path: Path) -> None:
    """load_attribute_index returns {} (not an error) when the file does not exist,
    mirroring app/color.py::load_color_index's graceful-degradation contract."""
    result = load_attribute_index(tmp_path / "does_not_exist.json")
    assert result == {}


def test_load_attribute_index_roundtrip(tmp_path: Path) -> None:
    """load_attribute_index reads back exactly what was written."""
    data = {
        "1": {
            "color": "black",
            "color_confidence": 0.12,
            "pattern": "solid",
            "pattern_confidence": 0.34,
            "fabric": "cotton",
            "fabric_confidence": 0.05,
            "occasion": "casual",
            "occasion_confidence": 0.21,
        }
    }
    p = tmp_path / "attributes.json"
    p.write_text(json.dumps(data))

    result = load_attribute_index(p)
    assert result == data


# ---------------------------------------------------------------------------
# Serve-path tests: GET /v1/{brand}/item/{item_id}/attributes
# ---------------------------------------------------------------------------

_SAMPLE_ENTRY = {
    "color": "black",
    "color_confidence": 0.12,
    "pattern": "solid",
    "pattern_confidence": 0.34,
    "fabric": "cotton",
    "fabric_confidence": 0.05,
    "occasion": "casual",
    "occasion_confidence": 0.21,
}


def _make_attributes_state(art_map: dict[int, dict], attributes: dict) -> MagicMock:
    """Build a MagicMock BrandState carrying only the fields item_attributes() reads."""
    state = MagicMock()
    state.api_key = "attrs-test-key"
    state.config.brand = "attrbrand"
    state.art_map = art_map
    state.attributes = attributes
    return state


def _make_attributes_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


def test_item_attributes_200_returns_precomputed_tags() -> None:
    art_map = {1: {"title": "Item 1"}}
    state = _make_attributes_state(art_map, {"1": _SAMPLE_ENTRY})
    registry = _make_attributes_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{state.config.brand}/item/1/attributes",
                headers={"X-Api-Key": "attrs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["item_id"] == "1"
    assert body["brand"] == state.config.brand
    assert body["color"] == "black"
    assert body["color_confidence"] == 0.12
    assert body["pattern"] == "solid"
    assert "request_id" in body
    # fabric/occasion are withheld from the API response entirely -- see
    # app/attributes.py::SERVED_ATTRIBUTES and app/api/schemas.py::ItemAttributesResponse.
    assert "fabric" not in body
    assert "fabric_confidence" not in body
    assert "occasion" not in body
    assert "occasion_confidence" not in body
    assert set(body["reliability"].keys()) == {"color", "pattern"}


def test_item_attributes_404_when_item_not_in_catalog() -> None:
    art_map = {1: {"title": "Item 1"}}
    state = _make_attributes_state(art_map, {"1": _SAMPLE_ENTRY})
    registry = _make_attributes_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{state.config.brand}/item/999/attributes",
                headers={"X-Api-Key": "attrs-test-key"},
            )

    assert resp.status_code == 404


def test_item_attributes_404_when_catalog_item_has_no_tags() -> None:
    """Item exists in the catalogue but was skipped at extraction time (e.g. no image)."""
    art_map = {1: {"title": "Item 1"}, 2: {"title": "Item 2"}}
    state = _make_attributes_state(art_map, {"1": _SAMPLE_ENTRY})
    registry = _make_attributes_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{state.config.brand}/item/2/attributes",
                headers={"X-Api-Key": "attrs-test-key"},
            )

    assert resp.status_code == 404


def test_item_attributes_503_when_unconfigured() -> None:
    """Brand has no attribute index at all (attributes_path unset or file missing)."""
    art_map = {1: {"title": "Item 1"}}
    state = _make_attributes_state(art_map, {})
    registry = _make_attributes_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{state.config.brand}/item/1/attributes",
                headers={"X-Api-Key": "attrs-test-key"},
            )

    assert resp.status_code == 503
