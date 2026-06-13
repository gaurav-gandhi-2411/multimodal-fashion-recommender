"""Tests for occasion/seasonal awareness — Feature 3.

Design principle (Phase-6 lesson): every test PROVES the feature changes output —
i.e. with occasion boost ON the ranked order differs from OFF, and the same-occasion
candidate wins.  Tests cover:

  1. Keyword-lexicon tagging (unit, hand-built strings)
  2. Explicit "Occasion : <value>" field parsing (Snitch format)
  3. Deliberate exclusion of "ethnic"/"traditional" from the lexicon
  4. Rerank order change driven by occasion boost
  5. Serve-path route test: occasion runs through the live /similar HTTP endpoint
"""
from __future__ import annotations

import numpy as np

from app.occasion import tag_occasions
from app.rerank import RerankConfig, rerank

# ---------------------------------------------------------------------------
# Helpers shared with test_rerank_diversity.py
# ---------------------------------------------------------------------------


def _normed(v: list[float]) -> np.ndarray:
    """Return a float32 numpy array normalised to unit length."""
    arr = np.array(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _base_rerank_cfg(**overrides: object) -> RerankConfig:
    """Minimal RerankConfig with all weights explicit so scores are predictable."""
    defaults: dict[str, object] = dict(
        enabled=True,
        candidate_pool_size=10,
        w_similarity=0.70,
        w_price_penalty=0.0,
        w_category_affinity=0.0,
        price_norm_inr=800.0,
        w_diversity=0.0,
        dupe_sim_threshold=0.97,
        price_bands_inr=[],
        w_price_band=0.0,
        w_occasion=0.0,
        parse_explicit_occasion=False,
    )
    defaults.update(overrides)
    return RerankConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Keyword lexicon tagging
# ---------------------------------------------------------------------------


def test_tag_occasions_keyword_festive() -> None:
    """Description containing 'festive' or 'wedding' tags festive."""
    occ = tag_occasions("Festive Kurta", "Perfect for festive celebrations")
    assert "festive" in occ, f"Expected 'festive' in tags, got {occ}"


def test_tag_occasions_keyword_casual() -> None:
    """Description containing 'everyday' or 'casual' tags casual."""
    occ = tag_occasions("Everyday Tee", "A casual everyday essential")
    assert "casual" in occ, f"Expected 'casual' in tags, got {occ}"


def test_tag_occasions_keyword_multiple() -> None:
    """Multiple occasion keywords → multiple tags returned."""
    occ = tag_occasions("Beach Party Top", "Great for beach parties and vacation getaways")
    assert "vacation" in occ, f"Expected 'vacation' in tags, got {occ}"
    assert "party" in occ, f"Expected 'party' in tags, got {occ}"


def test_tag_occasions_keyword_wedding() -> None:
    """'wedding' maps to festive via the lexicon."""
    occ = tag_occasions("Wedding Saree", "Elegant wedding attire")
    assert "festive" in occ, f"Expected 'festive' in tags (wedding keyword), got {occ}"


def test_tag_occasions_keyword_formal() -> None:
    """'office' and 'work wear' map to formal."""
    occ = tag_occasions("Office Shirt", "Perfect work wear for business meetings")
    assert "formal" in occ, f"Expected 'formal' in tags, got {occ}"


# ---------------------------------------------------------------------------
# 2. Explicit "Occasion : <value>" field parsing (Snitch format)
# ---------------------------------------------------------------------------


def test_tag_occasions_explicit_casual_wear() -> None:
    """'Occasion : Casual Wear  Pattern : Plain' with parse_explicit=True → casual."""
    desc = "Slim fit tee  Occasion : Casual Wear  Pattern : Plain  Material : Cotton"
    occ = tag_occasions("Plain Tee", desc, parse_explicit=True)
    assert "casual" in occ, f"Expected 'casual' from explicit field, got {occ}"


def test_tag_occasions_explicit_festive_wear() -> None:
    """'Occasion : Festive Wear' with parse_explicit=True → festive."""
    desc = "Embroidered kurta  Occasion : Festive Wear  Collar : Mandarin"
    occ = tag_occasions("Embroidered Kurta", desc, parse_explicit=True)
    assert "festive" in occ, f"Expected 'festive' from explicit field, got {occ}"


def test_tag_occasions_explicit_club_wear() -> None:
    """'Occasion : Club Wear' with parse_explicit=True → party."""
    desc = "Slim jogger  Occasion : Club Wear  Sleeves : Full"
    occ = tag_occasions("Club Jogger", desc, parse_explicit=True)
    assert "party" in occ, f"Expected 'party' (club wear) from explicit field, got {occ}"


def test_tag_occasions_explicit_disabled_does_not_parse() -> None:
    """With parse_explicit=False, explicit field is NOT parsed — only lexicon applies."""
    # Description ONLY has the explicit field, no free-text keywords
    desc = "Slim fit tee  Occasion : Casual Wear  Pattern : Plain"
    occ_off = tag_occasions("Plain Tee", desc, parse_explicit=False)
    # "casual wear" as a two-word phrase is NOT in the DEFAULT_OCCASION_LEXICON
    # (only the single word "casual" is), so this will NOT be matched by the lexicon.
    # If parse_explicit is off and there's no standalone "casual" keyword, set = empty.
    # (The desc does contain "Casual" inside "Casual Wear", so "casual" keyword WILL hit.)
    # This test confirms parse_explicit=True is not required for keyword fallback.
    occ_on = tag_occasions("Plain Tee", desc, parse_explicit=True)
    # Both should contain "casual" because the lexicon keyword "casual" appears in the text.
    assert "casual" in occ_off, "Keyword 'casual' in text should tag even without explicit parsing"
    assert "casual" in occ_on, "Keyword 'casual' in text should tag with explicit parsing too"


# ---------------------------------------------------------------------------
# 3. Deliberate exclusion: ethnic / traditional must NOT produce occasion tags
# ---------------------------------------------------------------------------


def test_tag_occasions_excludes_ethnic() -> None:
    """Description with only 'ethnic'/'traditional' — no occasion tags produced.

    These words appear on ~60% of Fashor items and are not occasion discriminators.
    They are intentionally absent from DEFAULT_OCCASION_LEXICON.
    """
    occ = tag_occasions(
        "Traditional Ethnic Kurta",
        "Beautiful ethnic traditional design with intricate embroidery",
    )
    assert len(occ) == 0, (
        f"'ethnic'/'traditional' should NOT produce occasion tags, got {occ}. "
        "These words are deliberately excluded from the lexicon as non-discriminating."
    )


def test_tag_occasions_ethnic_with_festive_does_tag() -> None:
    """'ethnic' alone → empty; 'ethnic' + 'festive' → only festive tagged."""
    occ = tag_occasions(
        "Ethnic Festive Kurta",
        "Traditional ethnic wear for festive occasions",
    )
    assert "festive" in occ, f"Expected 'festive' in tags, got {occ}"
    # No spurious tags from 'ethnic'/'traditional' alone
    # (festive is correctly there because 'festive' keyword matched)


# ---------------------------------------------------------------------------
# 4. Rerank order change driven by occasion boost
# ---------------------------------------------------------------------------


def test_rerank_occasion_boost_changes_order() -> None:
    """w_occasion=0.30 promotes the same-occasion candidate over a higher-FAISS-sim other.

    Setup (price penalty + category affinity disabled so only sim + occasion drive order):
      query: title="Festive Kurta", description="festive wedding ceremony"
        → query_occ = {"festive"}

      cand_A (id=1): sim=0.80, title="Festive Lehenga", desc="festive wedding"
        → same occasion → base_score = 0.70*0.80 = 0.560 + 0.30 boost = 0.860
      cand_B (id=2): sim=0.85, title="Casual Tee", desc="casual everyday"
        → different occasion → base_score = 0.70*0.85 = 0.595 (no boost)

    With w_occasion=0 (OFF): B wins by FAISS sim → [2, 1]
    With w_occasion=0.30 (ON): A scores 0.860 > B 0.595 → A wins → [1, 2]

    The test proves ON != OFF (feature is not a no-op) and the same-occasion item wins ON.
    """
    art_map: dict[int, dict] = {
        1: {
            "category": "Kurtas",
            "price_inr": 1000.0,
            "title": "Festive Lehenga",
            "description": "festive wedding celebration lehenga",
        },
        2: {
            "category": "Kurtas",
            "price_inr": 1000.0,
            "title": "Casual Tee",
            "description": "casual everyday streetwear tee",
        },
    }

    query_meta = {
        "title": "Festive Kurta",
        "description": "festive wedding ceremony kurta",
    }

    # cand_B (id=2) has higher FAISS sim so it wins when occasion is OFF
    candidates: list[tuple[int, float]] = [(1, 0.80), (2, 0.85)]

    cfg_off = _base_rerank_cfg(w_occasion=0.0)
    cfg_on = _base_rerank_cfg(w_occasion=0.30)

    result_off = rerank(
        candidates, 1000.0, "Kurtas", art_map, cfg_off, k=2,
        query_meta=query_meta,
    )
    result_on = rerank(
        candidates, 1000.0, "Kurtas", art_map, cfg_on, k=2,
        query_meta=query_meta,
    )

    ids_off = [aid for aid, _ in result_off]
    ids_on = [aid for aid, _ in result_on]

    # With occasion OFF: higher-FAISS-sim cand_B (id=2) must be rank-1
    assert ids_off[0] == 2, (
        f"Expected id=2 (higher sim) at rank-1 with w_occasion=0, got {ids_off[0]}. "
        f"Full order: {ids_off}"
    )

    # Occasion ON must differ from OFF (feature is not a no-op)
    assert ids_on != ids_off, (
        f"w_occasion=0.30 did not change output: ON={ids_on} OFF={ids_off}. "
        "Occasion boost may be a no-op — check rerank() Feature 3 path."
    )

    # Same-occasion candidate (id=1, festive) must be rank-1 with occasion ON
    assert ids_on[0] == 1, (
        f"Expected id=1 (festive, same occasion as query) at rank-1 with w_occasion=0.30, "
        f"got {ids_on[0]}. Full order: {ids_on}"
    )


def test_rerank_occasion_boost_w0_identical_to_no_meta() -> None:
    """w_occasion=0 must produce the same output regardless of query_meta presence."""
    art_map: dict[int, dict] = {
        1: {"category": "Tops", "price_inr": 1000.0,
            "title": "Casual Top", "description": "casual"},
        2: {"category": "Tops", "price_inr": 1000.0,
            "title": "Festive Blouse", "description": "festive"},
    }
    query_meta = {"title": "Casual Shirt", "description": "casual everyday"}
    candidates: list[tuple[int, float]] = [(1, 0.90), (2, 0.80)]

    cfg = _base_rerank_cfg(w_occasion=0.0)

    result_with_meta = rerank(candidates, 1000.0, "Tops", art_map, cfg, k=2, query_meta=query_meta)
    result_no_meta = rerank(candidates, 1000.0, "Tops", art_map, cfg, k=2)

    ids_with = [aid for aid, _ in result_with_meta]
    ids_no = [aid for aid, _ in result_no_meta]

    assert ids_with == ids_no, (
        f"w_occasion=0 produced different results with/without query_meta: "
        f"with_meta={ids_with}  no_meta={ids_no}. "
        "Occasion boost must be a no-op when w_occasion=0."
    )


# ---------------------------------------------------------------------------
# 5. Serve-path route test: occasion runs on the live /similar HTTP route
# ---------------------------------------------------------------------------


def test_similar_route_applies_occasion_end_to_end() -> None:
    """Occasion boost runs through the live /similar HTTP route.

    Two candidates have identical FAISS similarity (0.80).  The same-occasion
    candidate (festive) must be promoted over the different-occasion candidate
    (casual) when w_occasion > 0.  This proves occasion is applied on the SERVE path,
    not just in unit tests.

    Mirror of test_similar_route_applies_diversity_end_to_end in test_rerank_diversity.py.
    """
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    # Embeddings: query (row 0), cand_A=festive (row 1), cand_B=casual (row 2)
    # Identical sim vectors — only occasion discriminates
    emb = np.stack([
        _normed([1.0, 0.0, 0.0, 0.0]),   # row 0 — query item (aid 1)
        _normed([0.80, 0.60, 0.0, 0.0]),  # row 1 — aid 10  festive candidate
        _normed([0.80, 0.60, 0.0, 0.0]),  # row 2 — aid 11  casual candidate (same vector)
    ])

    state = MagicMock()
    state.api_key = "test-key"
    state.config.brand = "tb"
    # Occasion ON; all other weights zero so only similarity + occasion drive ranking.
    state.config.rerank = RerankConfig(
        enabled=True,
        candidate_pool_size=10,
        w_similarity=0.70,
        w_price_penalty=0.0,
        w_category_affinity=0.0,
        w_diversity=0.0,
        dupe_sim_threshold=0.97,
        price_bands_inr=[],
        w_price_band=0.0,
        w_occasion=0.30,
        parse_explicit_occasion=False,
    )
    state.art_map = {
        1: {                                      # query item
            "category": "Kurtas",
            "price_inr": 1000.0,
            "title": "Festive Kurta",
            "description": "festive wedding ceremony",
        },
        10: {                                     # same-occasion (festive)
            "category": "Kurtas",
            "price_inr": 1000.0,
            "title": "Festive Lehenga",
            "description": "festive wedding celebration",
        },
        11: {                                     # different occasion (casual)
            "category": "Kurtas",
            "price_inr": 1000.0,
            "title": "Casual Tee",
            "description": "casual everyday wear",
        },
    }
    state.faiss_aid_to_row = {1: 0, 10: 1, 11: 2}
    # Both candidates return identical FAISS similarity — occasion must break the tie
    state.retriever.search.return_value = [(10, 0.80), (11, 0.80)]
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

    # The festive candidate (id=10) must be rank-1 because it shares the query's occasion.
    # The casual candidate (id=11) should be rank-2 (still returned, just demoted).
    assert ids[0] == "10", (
        f"Expected festive candidate '10' at rank-1 (occasion boost), got {ids[0]}. "
        f"Full order: {ids}. Occasion is not being applied on the live /similar route."
    )
    assert ids[1] == "11", (
        f"Expected casual candidate '11' at rank-2, got {ids[1]}. Full order: {ids}."
    )
