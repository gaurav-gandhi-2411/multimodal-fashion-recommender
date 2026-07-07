"""tests/test_eval_visual_search_ab.py -- unit tests for the pure aggregation helpers in
scripts/eval_visual_search_ab.py (no model loading, no FAISS, no I/O).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_visual_search_ab import (  # noqa: E402
    QueryHit,
    overall_rate,
    per_category_rates,
    tshirt_shirt_confusion,
)


def _hit(category: str, top5_cats: list[str], self_retrieved: bool = True) -> QueryHit:
    """Build a QueryHit with cat_match_fraction/cat_hit derived like the real pass runners."""
    cat_match_fraction = sum(1 for c in top5_cats if c == category) / len(top5_cats)
    cat_hit = category in top5_cats
    return QueryHit(
        article_id=1,
        category=category,
        top5_cats=top5_cats,
        self_retrieved=self_retrieved,
        cat_match_fraction=cat_match_fraction,
        cat_hit=cat_hit,
    )


def test_overall_rate_empty_list_returns_zero() -> None:
    assert overall_rate([], "cat_hit") == 0.0


def test_overall_rate_mean_of_boolean_attribute() -> None:
    results = [
        _hit("Shirts", ["Shirts"] * 5),  # cat_match_fraction=1.0
        _hit("Shirts", ["Jeans"] * 5),  # cat_match_fraction=0.0
    ]
    assert overall_rate(results, "cat_match_fraction") == 0.5


def test_per_category_rates_filters_small_buckets() -> None:
    results = [
        _hit("Shirts", ["Shirts"] * 5) for _ in range(3)
    ] + [
        _hit("Jeans", ["Jeans"] * 5) for _ in range(2)  # below MIN_CATEGORY_QUERIES=3
    ]
    rates = per_category_rates(results, "cat_match_fraction", min_n=3)
    assert "Shirts" in rates
    assert rates["Shirts"] == (1.0, 3)
    assert "Jeans" not in rates  # only 2 queries, below threshold


def test_tshirt_shirt_confusion_counts_slots_correctly() -> None:
    # Two T-Shirt queries: one perfect (5/5 T-Shirts), one fully confused (5/5 Shirts).
    results = [
        _hit("T-Shirts", ["T-Shirts"] * 5),
        _hit("T-Shirts", ["Shirts"] * 5),
        _hit("Jeans", ["Jeans"] * 5),  # not a T-Shirt query -- must be excluded
    ]
    d = tshirt_shirt_confusion(results, tshirt_cat="T-Shirts", shirt_cat="Shirts")
    assert d["n_queries"] == 2
    assert d["n_slots"] == 10
    assert d["frac_tshirt"] == 0.5
    assert d["frac_shirt"] == 0.5
    assert d["frac_other"] == 0.0


def test_tshirt_shirt_confusion_no_matching_queries() -> None:
    results = [_hit("Jeans", ["Jeans"] * 5)]
    d = tshirt_shirt_confusion(results, tshirt_cat="T-Shirts", shirt_cat="Shirts")
    assert d["n_queries"] == 0
    assert d["frac_tshirt"] == 0.0
