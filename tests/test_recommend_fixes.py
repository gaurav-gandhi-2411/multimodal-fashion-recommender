"""tests/test_recommend_fixes.py

HTTP-path tests for the two /recommend regressions fixed in this PR:

  1. Self-exclusion: when item_id is provided the seed item must NOT appear in
     results (previously returned itself as rank-1 with score=1.0).

  2. Reranker applied: /recommend now calls the same category/price reranker as
     /similar, so overshirts/wrong-category items don't leak through for
     item-seed cold-start calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rerank import RerankConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEED_AID = 111
_OVERSHIRT_AID = 999
_SHIRT_AID_1 = 222
_SHIRT_AID_2 = 333

_ART_MAP = {
    _SEED_AID: {
        "article_id": _SEED_AID,
        "title": "Linen Slim Fit Shirt",
        "category": "Shirts",
        "price_inr": 1500.0,
    },
    _SHIRT_AID_1: {
        "article_id": _SHIRT_AID_1,
        "title": "Cotton Check Shirt",
        "category": "Shirts",
        "price_inr": 1200.0,
    },
    _SHIRT_AID_2: {
        "article_id": _SHIRT_AID_2,
        "title": "Striped Regular Fit Shirt",
        "category": "Shirts",
        "price_inr": 1100.0,
    },
    _OVERSHIRT_AID: {
        "article_id": _OVERSHIRT_AID,
        "title": "Relaxed Fit Grey Overshirt",
        "category": "Overshirt",
        "price_inr": 1600.0,
    },
}

# FAISS returns: seed first (score=1.0), then two shirts, then an overshirt.
# Before the fix recommend would include the seed item AND the overshirt.
_FAISS_RESULTS = [
    (_SEED_AID, 1.0),
    (_SHIRT_AID_1, 0.92),
    (_SHIRT_AID_2, 0.88),
    (_OVERSHIRT_AID, 0.85),
]


def _make_state_with_rerank(rerank_enabled: bool = True) -> MagicMock:
    state = MagicMock()
    state.api_key = "rec-test-key"
    state.config.brand = "testbrand"
    state.art_map = _ART_MAP
    state.user_history = None

    # Retriever returns seed + two shirts + one overshirt
    # Pool size is large enough to include all four candidates.
    state.retriever.search.return_value = _FAISS_RESULTS

    # Embeddings: dummy 256-d zero vectors (MMR diversity disabled via w_diversity=0)
    state.retriever.index.reconstruct.side_effect = (
        lambda row: np.zeros(256, dtype=np.float32)
    )

    rerank_cfg = RerankConfig(
        enabled=rerank_enabled,
        candidate_pool_size=20,
        w_similarity=0.70,
        w_price_penalty=0.15,
        w_category_affinity=0.15,
        price_norm_inr=500.0,
        equivalent_group_bonus=0.70,
        w_diversity=0.0,
    )
    state.config.rerank = rerank_cfg

    state.faiss_aid_to_row = {
        _SEED_AID: 0,
        _SHIRT_AID_1: 1,
        _SHIRT_AID_2: 2,
        _OVERSHIRT_AID: 3,
    }

    # item_embedding path for _get_item_embedding
    state.retriever.index.reconstruct.side_effect = (
        lambda row: np.ones(256, dtype=np.float32) / np.sqrt(256)
    )

    # model / device (needed by _get_user_embedding, not called here)
    state.device = __import__("torch").device("cpu")

    return state


def _make_registry(state: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.side_effect = lambda b: state if b == state.config.brand else None
    reg.brand_names.return_value = [state.config.brand]
    return reg


# ---------------------------------------------------------------------------
# Fix 1: self-exclusion
# ---------------------------------------------------------------------------


def test_recommend_seed_item_excluded_from_results() -> None:
    """The item_id seed must NOT appear in /recommend results (was rank-1 before fix)."""
    state = _make_state_with_rerank(rerank_enabled=False)
    registry = _make_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/recommend",
                json={"item_id": str(_SEED_AID), "k": 5},
                headers={"X-Api-Key": "rec-test-key"},
            )

    assert resp.status_code == 200, resp.text
    returned_ids = [r["item_id"] for r in resp.json()["results"]]
    assert str(_SEED_AID) not in returned_ids, (
        f"Seed item {_SEED_AID} must be excluded but appeared in results: {returned_ids}"
    )


def test_recommend_results_not_empty_after_exclusion() -> None:
    """After excluding the seed item, the remaining results should still be returned."""
    state = _make_state_with_rerank(rerank_enabled=False)
    registry = _make_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/recommend",
                json={"item_id": str(_SEED_AID), "k": 5},
                headers={"X-Api-Key": "rec-test-key"},
            )

    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) > 0, "No results returned after seed exclusion"
    # All returned items must exist in art_map
    for r in results:
        assert int(r["item_id"]) in _ART_MAP, f"Unknown item_id returned: {r['item_id']}"


# ---------------------------------------------------------------------------
# Fix 2: reranker applied — overshirt must not leak through
# ---------------------------------------------------------------------------


def test_recommend_reranker_filters_overshirt() -> None:
    """With reranker enabled, the Overshirt must not appear when query is a Shirt.

    The category_affinity component of the reranker penalises items whose
    category doesn't match the query's, which pushes the Overshirt below the
    shirts.  This mirrors the behaviour that /similar already had.
    """
    state = _make_state_with_rerank(rerank_enabled=True)
    registry = _make_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/recommend",
                json={"item_id": str(_SEED_AID), "k": 2},
                headers={"X-Api-Key": "rec-test-key"},
            )

    assert resp.status_code == 200, resp.text
    returned_ids = [r["item_id"] for r in resp.json()["results"]]

    # Overshirt should be downranked below the two shirts, so not in top-2.
    assert str(_OVERSHIRT_AID) not in returned_ids, (
        f"Overshirt ({_OVERSHIRT_AID}) leaked into top-2 despite reranker. "
        f"Results: {returned_ids}"
    )


def test_recommend_reranker_keeps_same_category_items() -> None:
    """Reranker keeps same-category shirts in the results after filtering seed + overshirt."""
    state = _make_state_with_rerank(rerank_enabled=True)
    registry = _make_registry(state)

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{state.config.brand}/recommend",
                json={"item_id": str(_SEED_AID), "k": 5},
                headers={"X-Api-Key": "rec-test-key"},
            )

    assert resp.status_code == 200, resp.text
    returned_ids = [r["item_id"] for r in resp.json()["results"]]

    assert str(_SHIRT_AID_1) in returned_ids or str(_SHIRT_AID_2) in returned_ids, (
        f"Expected at least one same-category shirt in results. Got: {returned_ids}"
    )
