"""Tests for Complete-the-Look (outfit completion).

Guardrail (Phase 6 lesson): prove the feature CHANGES output and is not a no-op.
These use hand-built numpy vectors + CompleteConfig — no FAISS/torch/data needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.complete import CompleteConfig, OutfitSlot, build_slot_index, complete_the_look


def _unit(vec: list[float]) -> np.ndarray:
    a = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


@pytest.fixture()
def menswear_config() -> CompleteConfig:
    """TOP / BOTTOM / LAYER slots; a TOP query completes with BOTTOM + LAYER."""
    return CompleteConfig(
        enabled=True,
        per_slot=2,
        max_items=6,
        w_style=0.7,
        w_price=0.3,
        price_norm_inr=500.0,
        slots=[
            OutfitSlot(name="TOP", categories=["Shirts", "T-Shirts"]),
            OutfitSlot(name="BOTTOM", categories=["Jeans", "Trousers"]),
            OutfitSlot(name="LAYER", categories=["Jackets", "Sweaters"]),
        ],
        complements={"TOP": ["BOTTOM", "LAYER"], "BOTTOM": ["TOP", "LAYER"]},
    )


def test_returns_only_complementary_categories(menswear_config: CompleteConfig) -> None:
    """A TOP query must return ZERO same-slot (TOP) items — proves it is NOT /similar."""
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [
        # same slot as query (TOP) — must be excluded entirely
        ("t1", "Shirts", 1000.0, _unit([1.0, 0.0, 0.0])),
        ("t2", "T-Shirts", 1000.0, _unit([0.99, 0.1, 0.0])),
        # complementary
        ("b1", "Jeans", 1000.0, _unit([0.8, 0.2, 0.0])),
        ("b2", "Trousers", 1100.0, _unit([0.7, 0.3, 0.0])),
        ("l1", "Jackets", 1050.0, _unit([0.6, 0.4, 0.0])),
    ]
    results = complete_the_look("Shirts", q_emb, 1000.0, candidates, menswear_config)
    returned_ids = {aid for aid, _, _ in results}
    assert returned_ids and returned_ids.isdisjoint({"t1", "t2"}), (
        f"Complete-the-Look returned same-category items {returned_ids & {'t1', 't2'}}; "
        "it must only return COMPLEMENTARY categories."
    )
    slot_index = build_slot_index(menswear_config)
    for aid, _, slot in results:
        cat = next(c for i, c, _, _ in candidates if i == aid)
        assert slot in ("BOTTOM", "LAYER")
        assert slot_index[cat] == slot


def test_spans_multiple_slots(menswear_config: CompleteConfig) -> None:
    """With BOTTOM and LAYER candidates available, the outfit covers >1 slot."""
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [
        ("b1", "Jeans", 1000.0, _unit([0.9, 0.1, 0.0])),
        ("b2", "Trousers", 1000.0, _unit([0.85, 0.15, 0.0])),
        ("b3", "Jeans", 1000.0, _unit([0.8, 0.2, 0.0])),
        ("l1", "Jackets", 1000.0, _unit([0.7, 0.3, 0.0])),
        ("l2", "Sweaters", 1000.0, _unit([0.6, 0.4, 0.0])),
    ]
    results = complete_the_look("Shirts", q_emb, 1000.0, candidates, menswear_config)
    slots = {slot for _, _, slot in results}
    assert slots == {"BOTTOM", "LAYER"}, f"Expected outfit to span BOTTOM+LAYER, got {slots}"
    # per_slot=2 cap → at most 2 BOTTOM despite 3 available
    bottoms = [aid for aid, _, slot in results if slot == "BOTTOM"]
    assert len(bottoms) <= 2


def test_style_and_price_scoring_orders(menswear_config: CompleteConfig) -> None:
    """Within a slot, the higher-cosine + closer-price candidate ranks first (not a no-op)."""
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [
        # far in style AND price
        ("b_far", "Jeans", 5000.0, _unit([0.0, 1.0, 0.0])),
        # close in style AND price — should win
        ("b_close", "Jeans", 1000.0, _unit([0.98, 0.02, 0.0])),
    ]
    results = complete_the_look("Shirts", q_emb, 1000.0, candidates, menswear_config)
    assert results[0][0] == "b_close", (
        f"Expected the style+price-coherent candidate first, got {results[0][0]}. "
        "Scoring may be a no-op."
    )


def test_disabled_returns_empty(menswear_config: CompleteConfig) -> None:
    cfg = menswear_config.model_copy(update={"enabled": False})
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [("b1", "Jeans", 1000.0, _unit([0.9, 0.1, 0.0]))]
    assert complete_the_look("Shirts", q_emb, 1000.0, candidates, cfg) == []


def test_unknown_query_slot_returns_empty(menswear_config: CompleteConfig) -> None:
    """A query category that is not in any slot (e.g. an Accessory) yields no outfit."""
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [("b1", "Jeans", 1000.0, _unit([0.9, 0.1, 0.0]))]
    assert complete_the_look("Belts", q_emb, 1000.0, candidates, menswear_config) == []


def test_slot_with_no_complements_returns_empty(menswear_config: CompleteConfig) -> None:
    """LAYER has no complements defined in this fixture → empty."""
    q_emb = _unit([1.0, 0.0, 0.0])
    candidates = [("b1", "Jeans", 1000.0, _unit([0.9, 0.1, 0.0]))]
    assert complete_the_look("Jackets", q_emb, 1000.0, candidates, menswear_config) == []


# ---------------------------------------------------------------------------
# Route-level: prove the LIVE /complete endpoint executes the feature end-to-end
# ---------------------------------------------------------------------------


def test_complete_route_returns_complementary_outfit() -> None:
    """GET /v1/{brand}/item/{id}/complete returns complementary-category items.

    Builds a minimal real-data brand state (real CompleteConfig + embedding matrix)
    so the assertion exercises the actual route handler, auth, and response schema —
    not just the pure scoring function.
    """
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    emb = np.zeros((3, 4), dtype=np.float32)
    emb[0] = _unit([1.0, 0.0, 0.0, 0.0])  # item 1 — Shirts (query, TOP)
    emb[1] = _unit([0.8, 0.2, 0.0, 0.0])  # item 2 — Jeans (BOTTOM)
    emb[2] = _unit([0.6, 0.4, 0.0, 0.0])  # item 3 — Jackets (LAYER)

    cfg = CompleteConfig(
        enabled=True,
        per_slot=2,
        max_items=6,
        slots=[
            OutfitSlot(name="TOP", categories=["Shirts"]),
            OutfitSlot(name="BOTTOM", categories=["Jeans"]),
            OutfitSlot(name="LAYER", categories=["Jackets"]),
        ],
        complements={"TOP": ["BOTTOM", "LAYER"]},
    )

    state = MagicMock()
    state.api_key = "test-key"
    state.config.brand = "tb"
    state.config.complete = cfg
    state.art_map = {
        1: {"category": "Shirts", "price_inr": 1000.0, "pdp_url": None},
        2: {"category": "Jeans", "price_inr": 1100.0, "pdp_url": "https://x/2"},
        3: {"category": "Jackets", "price_inr": 1050.0, "pdp_url": None},
    }
    state.faiss_aid_to_row = {1: 0, 2: 1, 3: 2}
    state.item_embeddings = emb
    state.retriever.index.reconstruct.side_effect = lambda row: emb[row]

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == "tb" else None
    registry.brand_names.return_value = ["tb"]

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app) as client:
            resp = client.get("/v1/tb/item/1/complete", headers={"X-Api-Key": "test-key"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is True
    cats = {item["item_id"]: item["slot"] for item in body["results"]}
    assert cats, "expected a non-empty outfit"
    # query item (1, Shirts/TOP) must NOT appear; results are complementary only
    assert "1" not in cats
    assert set(cats.keys()) == {"2", "3"}
    assert sorted(body["slots_covered"]) == ["BOTTOM", "LAYER"]

    # SERVE-PATH PROOF (Phase-6 lesson): assert on the actual garment CATEGORY each
    # returned item carries — NONE may equal the query's category ("Shirts"). This
    # proves /complete returns COMPLEMENTARY items, not same-category /similar items,
    # on the live HTTP route — not only in the eval harness.
    id_to_category = {str(aid): meta["category"] for aid, meta in state.art_map.items()}
    returned_categories = {id_to_category[item_id] for item_id in cats}
    assert "Shirts" not in returned_categories, (
        f"/complete returned a same-category (Shirts) item: {returned_categories}. "
        "Complete-the-Look must return complementary categories on the serve path."
    )
    assert returned_categories == {"Jeans", "Jackets"}


def test_complete_route_disabled_brand_returns_enabled_false() -> None:
    """A brand with complete.enabled=false returns 200 with enabled=False, empty results."""
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    emb = np.zeros((1, 4), dtype=np.float32)
    emb[0] = _unit([1.0, 0.0, 0.0, 0.0])

    state = MagicMock()
    state.api_key = "test-key"
    state.config.brand = "tb"
    state.config.complete = CompleteConfig(enabled=False)
    state.art_map = {1: {"category": "Kurtas", "price_inr": 1000.0}}
    state.faiss_aid_to_row = {1: 0}
    state.item_embeddings = emb
    state.retriever.index.reconstruct.side_effect = lambda row: emb[row]

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == "tb" else None
    registry.brand_names.return_value = ["tb"]

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app) as client:
            resp = client.get("/v1/tb/item/1/complete", headers={"X-Api-Key": "test-key"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is False
    assert body["results"] == []
