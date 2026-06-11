"""Regression test: rerank must not be a no-op when candidate IDs are strings.

FaissRetriever.search() returns str article IDs (from article_ids.pkl).
Before the Phase 6 fix, art_map had int keys and art_map.get(str_id, {}) always
returned {} -- making every rerank call a silent no-op identical to raw FAISS order.

This file asserts:
  1. The key-normalization fix allows art_map lookups to succeed with str candidates.
  2. For a query where price/category signals clearly favour a lower-FAISS-ranked item,
     rerank produces a different top-1 than raw FAISS order.
  3. The pre-fix behaviour (no normalization) would have preserved FAISS order,
     demonstrating what the bug looked like.
"""
from __future__ import annotations

import pytest

from app.rerank import CategoryAffinityMap, RerankConfig, rerank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> RerankConfig:
    """Minimal rerank config: two categories in one equivalence group."""
    return RerankConfig(
        enabled=True,
        candidate_pool_size=10,
        w_similarity=0.70,
        w_price_penalty=0.15,
        w_category_affinity=0.15,
        price_norm_inr=800.0,
    )


@pytest.fixture()
def art_map_int_keys() -> dict[int, dict]:
    """art_map as built by registry.py: int keys, realistic metadata."""
    return {
        1: {"title": "Off-category item at far price", "category": "Dresses", "price_inr": 5000.0},
        2: {"title": "Same-category item at close price", "category": "Kurtas", "price_inr": 1100.0},
        3: {"title": "Same-category item mid price", "category": "Kurtas", "price_inr": 1300.0},
    }


# ---------------------------------------------------------------------------
# 1. Key normalization: str candidates resolve int art_map entries
# ---------------------------------------------------------------------------

def test_str_candidate_resolves_int_art_map(config: RerankConfig, art_map_int_keys: dict[int, dict]) -> None:
    """art_map.get(str_id, {}) returns {} pre-fix; after fix, non-empty metadata is returned."""
    str_candidates: list[tuple[str, float]] = [("1", 0.90), ("2", 0.85)]

    # Pre-fix: no normalisation -- every lookup returns {}
    for art_id, _ in str_candidates:
        assert art_map_int_keys.get(art_id, {}) == {}, (  # type: ignore[arg-type]
            "str key should not match int art_map -- this confirms the original bug condition"
        )

    # Post-fix: the normalisation inside rerank() makes lookups succeed
    # We verify this indirectly by calling rerank() and checking that scores differ
    # (if metadata were empty, price_penalty=0 and cat_affinity=0 for all items,
    #  so scores would be proportional to sim and FAISS order would be preserved)
    results = rerank(str_candidates, 1000.0, "Kurtas", art_map_int_keys, config, k=2)
    ids = [str(art_id) for art_id, _ in results]

    # Item "2" (same cat, close price) should beat item "1" (wrong cat, far price)
    # despite item "1" having higher FAISS similarity
    assert ids[0] == "2", (
        f"Expected item '2' (same category, close price) at rank 1, got {ids[0]}. "
        "Metadata lookup may have failed -- check str/int key normalization in rerank()."
    )


# ---------------------------------------------------------------------------
# 2. Rerank order differs from raw FAISS order for a known price+category signal
# ---------------------------------------------------------------------------

def test_rerank_differs_from_faiss_order(config: RerankConfig, art_map_int_keys: dict[int, dict]) -> None:
    """For str candidates with int art_map keys, reranked top-1 != FAISS top-1.

    Query: Kurtas, Rs.1000
    FAISS order by sim:  item "1" (0.95) > item "2" (0.90) > item "3" (0.85)
    Rerank signals:
      item "1": cat=Dresses (affinity=0), price=5000 (penalty=min(1,4000/800)=1.0)
                score = 0.70*0.95 - 0.15*1.0 + 0.15*0.0 = 0.515
      item "2": cat=Kurtas (affinity=1.0), price=1100 (penalty=min(1,100/800)=0.125)
                score = 0.70*0.90 - 0.15*0.125 + 0.15*1.0 = 0.7613
      item "3": cat=Kurtas (affinity=1.0), price=1300 (penalty=min(1,300/800)=0.375)
                score = 0.70*0.85 - 0.15*0.375 + 0.15*1.0 = 0.6888
    Reranked order: item "2" > item "3" > item "1"
    """
    # Candidates as str (simulating FaissRetriever.search output)
    str_candidates: list[tuple[str, float]] = [
        ("1", 0.95),  # FAISS rank 1 -- wrong category, far price
        ("2", 0.90),  # FAISS rank 2 -- right category, close price
        ("3", 0.85),  # FAISS rank 3 -- right category, mid price
    ]
    raw_top1 = str_candidates[0][0]  # "1" by FAISS order

    results = rerank(str_candidates, 1000.0, "Kurtas", art_map_int_keys, config, k=3)
    reranked_ids = [str(art_id) for art_id, _ in results]

    assert reranked_ids[0] != raw_top1, (
        f"Reranked top-1 ({reranked_ids[0]!r}) == FAISS top-1 ({raw_top1!r}): "
        "rerank may be a no-op. Check str/int key normalization in rerank()."
    )
    assert reranked_ids[0] == "2", (
        f"Expected item '2' at reranked rank 1, got {reranked_ids[0]!r}. "
        f"Full reranked order: {reranked_ids}"
    )


# ---------------------------------------------------------------------------
# 3. Rerank is idempotent for already-correct int candidates (non-regression)
# ---------------------------------------------------------------------------

def test_rerank_works_for_int_candidates_too(config: RerankConfig, art_map_int_keys: dict[int, dict]) -> None:
    """Normalization must not break the path where candidates are already int."""
    int_candidates: list[tuple[int, float]] = [
        (1, 0.95),
        (2, 0.90),
        (3, 0.85),
    ]
    results = rerank(int_candidates, 1000.0, "Kurtas", art_map_int_keys, config, k=3)
    reranked_ids = [int(art_id) for art_id, _ in results]

    assert reranked_ids[0] == 2, (
        f"Int candidates: expected item 2 at rank 1, got {reranked_ids[0]}. "
        f"Full order: {reranked_ids}"
    )
