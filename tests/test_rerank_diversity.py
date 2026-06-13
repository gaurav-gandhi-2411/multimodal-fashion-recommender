"""Tests for MMR diversity (Feature 1) and price-band coherence (Feature 2) in rerank().

Design principle: every test PROVES the feature changes output — i.e. the reranked order
with the feature ON differs from the order with the feature OFF.  No FAISS/torch/data
fixtures needed; all inputs are small numpy arrays constructed by hand.
"""
from __future__ import annotations

import numpy as np

from app.rerank import RerankConfig, rerank

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normed(v: list[float]) -> np.ndarray:
    """Return a float32 numpy array normalised to unit length."""
    arr = np.array(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _base_config(**overrides: object) -> RerankConfig:
    """Minimal RerankConfig with all weights explicit so scores are predictable."""
    defaults: dict[str, object] = dict(
        enabled=True,
        candidate_pool_size=10,
        w_similarity=0.70,
        w_price_penalty=0.0,   # disable price penalty to isolate signal under test
        w_category_affinity=0.0,
        price_norm_inr=800.0,
        w_diversity=0.0,
        dupe_sim_threshold=0.97,
        price_bands_inr=[],
        w_price_band=0.0,
    )
    defaults.update(overrides)
    return RerankConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shared art_map: all items have same category + price so base score = w_sim * sim only
# ---------------------------------------------------------------------------

_ART_MAP: dict[int, dict] = {
    1: {"category": "Tops", "price_inr": 1000.0},
    2: {"category": "Tops", "price_inr": 1000.0},
    3: {"category": "Tops", "price_inr": 1000.0},
    4: {"category": "Tops", "price_inr": 1000.0},
}


# ---------------------------------------------------------------------------
# Test 1 — MMR demotes a near-duplicate
# ---------------------------------------------------------------------------

def test_diversity_demotes_near_duplicate() -> None:
    """With w_diversity=0.5 the second near-dup is pushed out of top-3.

    Candidates:
      id=1  sim=0.90   embedding = [1, 0, 0] (reference direction)
      id=2  sim=0.89   embedding ≈ [0.9999, 0.01, 0] (cosine to id=1 ≈ 0.9999 → redundant)
      id=3  sim=0.80   embedding = [0, 1, 0]  (orthogonal to id=1 — diverse)
      id=4  sim=0.78   embedding = [0, 0, 1]  (orthogonal to both — diverse)

    Diversity OFF order (by base_score = 0.7*sim):
        id=1 (0.63) > id=2 (0.623) > id=3 (0.56) > id=4 (0.546)
        top-3 = [1, 2, 3]

    Diversity ON (w_diversity=0.5):
      Round 1: all max_cos=0 → pick id=1 (highest base_score)
      Round 2: id=2 MMR = 0.623 - 0.5*0.9999 ≈ 0.123
               id=3 MMR = 0.56  - 0.5*0.0    = 0.56   ← higher
               id=4 MMR = 0.546 - 0.5*0.0    = 0.546
               → pick id=3
      Round 3: id=2 MMR ≈ 0.623 - 0.5*0.9999 ≈ 0.123
               id=4 MMR = 0.546 - 0.5*0.0    = 0.546  ← higher
               → pick id=4
        top-3 = [1, 3, 4]   — id=2 (near-dup) demoted to rank 4
    """
    # id=2 is near-identical to id=1
    v1 = _normed([1.0, 0.0, 0.0])
    v2 = _normed([0.9999, 0.0141, 0.0])   # cosine to v1 ≈ 0.9999
    v3 = _normed([0.0, 1.0, 0.0])
    v4 = _normed([0.0, 0.0, 1.0])

    candidates = [(1, 0.90), (2, 0.89), (3, 0.80), (4, 0.78)]
    embeddings = {1: v1, 2: v2, 3: v3, 4: v4}

    config_off = _base_config(w_diversity=0.0)
    config_on  = _base_config(w_diversity=0.5, dupe_sim_threshold=0.97)

    result_off = rerank(
        candidates, 1000.0, "Tops", _ART_MAP, config_off, k=3, embeddings=embeddings
    )
    result_on = rerank(
        candidates, 1000.0, "Tops", _ART_MAP, config_on, k=3, embeddings=embeddings
    )

    ids_off = [aid for aid, _ in result_off]
    ids_on  = [aid for aid, _ in result_on]

    # Sanity: diversity OFF keeps FAISS-sim order → id=2 is rank-2
    assert ids_off == [1, 2, 3], f"diversity OFF unexpected order: {ids_off}"

    # Diversity ON must differ from diversity OFF
    assert ids_on != ids_off, (
        f"MMR did not change output: diversity ON order {ids_on} == diversity OFF order {ids_off}. "
        "Feature may be a no-op — check _mmr_select()."
    )

    # id=2 (near-dup of id=1) must NOT be in top-3 with diversity ON
    assert 2 not in ids_on, (
        f"Near-duplicate id=2 still in top-3 with diversity ON: {ids_on}. "
        "MMR is not penalising the redundant candidate."
    )

    # The diverse items (3, 4) must fill the slots instead
    assert set(ids_on) == {1, 3, 4}, (
        f"Expected top-3 to be {{1, 3, 4}} with diversity ON, got {ids_on}"
    )


# ---------------------------------------------------------------------------
# Test 2 — w_diversity=0 (or embeddings=None) is backward-compatible
# ---------------------------------------------------------------------------

def test_diversity_off_is_backward_compatible() -> None:
    """w_diversity=0 must produce the same output whether embeddings are given or not."""
    v1 = _normed([1.0, 0.0, 0.0])
    v2 = _normed([0.9999, 0.0141, 0.0])
    v3 = _normed([0.0, 1.0, 0.0])

    candidates = [(1, 0.90), (2, 0.89), (3, 0.80)]
    embeddings = {1: v1, 2: v2, 3: v3}

    config = _base_config(w_diversity=0.0)

    result_with_emb = rerank(
        candidates, 1000.0, "Tops", _ART_MAP, config, k=3, embeddings=embeddings
    )
    result_without_emb = rerank(
        candidates, 1000.0, "Tops", _ART_MAP, config, k=3, embeddings=None
    )
    result_no_kwarg = rerank(candidates, 1000.0, "Tops", _ART_MAP, config, k=3)

    ids_with    = [aid for aid, _ in result_with_emb]
    ids_without = [aid for aid, _ in result_without_emb]
    ids_no_kw   = [aid for aid, _ in result_no_kwarg]

    assert ids_with == ids_without == ids_no_kw, (
        f"w_diversity=0 produced different results depending on embeddings kwarg: "
        f"with={ids_with}  without={ids_without}  no_kwarg={ids_no_kw}"
    )


# ---------------------------------------------------------------------------
# Test 3 — price-band bonus changes ranking order
# ---------------------------------------------------------------------------

def test_price_band_bonus_changes_order() -> None:
    """A same-band candidate beats an out-of-band candidate when w_price_band is high enough.

    Setup (price penalty and category affinity disabled):
      query_price = ₹800  → band 1 (₹600-₹1200)

      id=10  sim=0.90  price=850   → band 1 (same band as query)
             base_score without bonus = 0.70 * 0.90 = 0.630
             base_score with    bonus = 0.630 + 0.20 = 0.830

      id=11  sim=0.92  price=1500  → band 2 (out of band)
             base_score = 0.70 * 0.92 = 0.644  (no bonus)

    Without band bonus: id=11 (0.644) > id=10 (0.630)
    With    band bonus: id=10 (0.830) > id=11 (0.644)
    """
    art_map_local: dict[int, dict] = {
        10: {"category": "Tops", "price_inr": 850.0},
        11: {"category": "Tops", "price_inr": 1500.0},
    }

    candidates = [(10, 0.90), (11, 0.92)]

    config_off = _base_config(price_bands_inr=[], w_price_band=0.0)
    config_on  = _base_config(
        price_bands_inr=[600.0, 1200.0, 2000.0],
        w_price_band=0.20,   # deliberately large to guarantee order flip
    )

    result_off = rerank(candidates, 800.0, "Tops", art_map_local, config_off, k=2)
    result_on  = rerank(candidates, 800.0, "Tops", art_map_local, config_on,  k=2)

    ids_off = [aid for aid, _ in result_off]
    ids_on  = [aid for aid, _ in result_on]

    # Without bonus: higher sim wins → id=11 is rank-1
    assert ids_off[0] == 11, (
        f"Expected id=11 at rank-1 with band bonus OFF, got {ids_off[0]}. "
        f"Full order: {ids_off}"
    )

    # With bonus: same-band id=10 must be promoted to rank-1
    assert ids_on[0] == 10, (
        f"Expected id=10 at rank-1 with band bonus ON (same band as query), got {ids_on[0]}. "
        f"Full order: {ids_on}. "
        "Price-band bonus may not be reaching base_score — check rerank() Feature 2 path."
    )

    # Output must differ between the two configs
    assert ids_on != ids_off, (
        f"Price-band bonus did not change output: ON={ids_on} OFF={ids_off}. "
        "Feature may be a no-op."
    )


# ---------------------------------------------------------------------------
# Test 4 — SERVE-PATH PROOF: diversity runs through the live /similar HTTP route
# ---------------------------------------------------------------------------
# This is the Phase-6-lesson guard: the eval and the rerank() unit tests prove the
# FUNCTION works, but the no-op bug was a SERVE-path divergence. This test calls
# GET /v1/{brand}/item/{id}/similar via the FastAPI test client and asserts the HTTP
# response reflects MMR — a near-duplicate that raw FAISS ranks ABOVE a diverse item
# is demoted out of the top-k in the actual API response.


def test_similar_route_applies_diversity_end_to_end() -> None:
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    # Embedding rows: query(0); A(1) and A2(2) are near-duplicates (cos≈0.999);
    # B(3) is diverse. Raw FAISS sim order: A(0.80) > A2(0.79) > B(0.70).
    emb = np.stack([
        _normed([1.0, 0.0, 0.0, 0.0]),   # row 0 — query item (aid 1)
        _normed([0.80, 0.60, 0.0, 0.0]),  # row 1 — A   (aid 10)
        _normed([0.79, 0.61, 0.05, 0.0]),  # row 2 — A2  (aid 11) near-dup of A
        _normed([0.70, 0.0, 0.714, 0.0]),  # row 3 — B   (aid 20) diverse
    ])

    state = MagicMock()
    state.api_key = "test-key"
    state.config.brand = "tb"
    # Diversity ON; price/category disabled so only sim + MMR drive the order.
    state.config.rerank = RerankConfig(
        enabled=True, candidate_pool_size=10,
        w_similarity=0.70, w_price_penalty=0.0, w_category_affinity=0.0,
        w_diversity=0.5, dupe_sim_threshold=0.92, price_bands_inr=[], w_price_band=0.0,
    )
    state.art_map = {
        10: {"category": "X", "price_inr": 1000.0},
        11: {"category": "X", "price_inr": 1000.0},
        20: {"category": "X", "price_inr": 1000.0},
    }
    state.faiss_aid_to_row = {1: 0, 10: 1, 11: 2, 20: 3}
    # Raw FAISS order ranks the near-dup A2 (0.79) ABOVE the diverse B (0.70).
    state.retriever.search.return_value = [(10, 0.80), (11, 0.79), (20, 0.70)]
    state.retriever.index.reconstruct.side_effect = lambda row: emb[row]

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == "tb" else None
    registry.brand_names.return_value = ["tb"]

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app

        with TestClient(app) as client:
            resp = client.get("/v1/tb/item/1/similar?k=2", headers={"X-Api-Key": "test-key"})

    assert resp.status_code == 200, resp.text
    ids = [r["item_id"] for r in resp.json()["results"]]
    # Raw FAISS top-2 would be [10, 11] (A + its near-dup). MMR must demote the near-dup
    # 11 and promote the diverse item 20 — proving diversity ran on the SERVE path.
    assert ids == ["10", "20"], (
        f"Expected MMR to return ['10','20'] (near-dup '11' demoted for diverse '20'), "
        f"got {ids}. Diversity is not being applied on the live /similar route."
    )
    assert "11" not in ids, "Near-duplicate '11' survived in the HTTP response — MMR not applied on serve path."
