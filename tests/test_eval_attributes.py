"""tests/test_eval_attributes.py -- unit tests for the pure cross-validation helpers in
scripts/eval_attributes.py (no model loading, no FAISS, no network I/O).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_attributes import (  # noqa: E402
    CategoryEvalResult,
    OccasionEvalResult,
    build_spotcheck_sample,
    confidence_gap,
    eval_category_text_xval,
    eval_occasion_text_xval,
    find_text_labels,
    majority_baseline_accuracy,
    majority_predicted_label,
    pool_category_results,
    pool_occasion_results,
    unambiguous_text_label,
)

_COLOR_LABELS = ["black", "white", "navy", "multicolor"]


def _items_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal catalogue DataFrame with the columns eval_attributes.py reads."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# find_text_labels / unambiguous_text_label
# ---------------------------------------------------------------------------


def test_find_text_labels_single_match() -> None:
    assert find_text_labels("A sharp black shirt", _COLOR_LABELS) == ["black"]


def test_find_text_labels_multi_word_phrase_matches_as_phrase() -> None:
    labels = ["polka dot", "solid", "striped"]
    assert find_text_labels("A polka dot dress", labels) == ["polka dot"]
    # "dot" alone must not falsely satisfy "polka dot" without "polka" present
    assert find_text_labels("a dotted texture", labels) == []


def test_find_text_labels_no_match() -> None:
    assert find_text_labels("A plain grey trouser", _COLOR_LABELS) == []


def test_find_text_labels_ambiguous_multiple_matches() -> None:
    assert find_text_labels("black and white striped top", _COLOR_LABELS) == ["black", "white"]


def test_unambiguous_text_label_returns_label_when_single_match() -> None:
    assert unambiguous_text_label("a navy blazer", _COLOR_LABELS) == "navy"


def test_unambiguous_text_label_returns_none_when_ambiguous() -> None:
    assert unambiguous_text_label("black and white striped top", _COLOR_LABELS) is None


def test_unambiguous_text_label_returns_none_when_no_match() -> None:
    assert unambiguous_text_label("a plain grey trouser", _COLOR_LABELS) is None


# ---------------------------------------------------------------------------
# eval_category_text_xval
# ---------------------------------------------------------------------------


def test_eval_category_text_xval_coverage_and_accuracy() -> None:
    items = _items_df(
        [
            {"article_id": 1, "title": "black shirt", "description": ""},  # unambiguous, correct
            {"article_id": 2, "title": "white shirt", "description": ""},  # unambiguous, wrong
            {"article_id": 3, "title": "black and white shirt", "description": ""},  # ambiguous
            {"article_id": 4, "title": "plain shirt", "description": ""},  # no text signal
        ]
    )
    attr_index = {
        "1": {"color": "black", "color_confidence": 0.9},
        "2": {"color": "navy", "color_confidence": 0.1},  # wrong vs text-derived "white"
        "3": {"color": "black", "color_confidence": 0.5},
        "4": {"color": "black", "color_confidence": 0.2},
    }
    result = eval_category_text_xval(items, attr_index, "color")

    assert result.n_total == 4
    assert result.n_unambiguous == 2  # items 1, 2
    assert result.n_evaluated == 2
    assert result.n_correct == 1  # only item 1
    assert result.coverage == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)
    assert result.correct_confidences == [0.9]
    assert result.incorrect_confidences == [0.1]


def test_eval_category_text_xval_skips_items_without_prediction() -> None:
    items = _items_df([{"article_id": 1, "title": "black shirt", "description": ""}])
    result = eval_category_text_xval(items, {}, "color")  # no prediction for item 1

    assert result.n_total == 1
    assert result.n_unambiguous == 1
    assert result.n_evaluated == 0
    assert result.n_correct == 0
    assert result.accuracy == 0.0  # no ZeroDivisionError


def test_pool_category_results_sums_raw_counts_not_rates() -> None:
    """Pooling must sum counts (not average per-brand rates) so uneven brand sizes don't
    distort the pooled number."""
    a = CategoryEvalResult(
        category="color", n_total=100, n_unambiguous=50, n_evaluated=50, n_correct=25
    )  # 50% accuracy on 50 items
    b = CategoryEvalResult(
        category="color", n_total=10, n_unambiguous=10, n_evaluated=10, n_correct=10
    )  # 100% accuracy on 10 items
    pooled = pool_category_results([a, b], "color")

    assert pooled.n_total == 110
    assert pooled.n_unambiguous == 60
    assert pooled.n_evaluated == 60
    assert pooled.n_correct == 35
    # Pooled accuracy is count-weighted (35/60), not the naive average of 50% and 100% (75%).
    assert pooled.accuracy == pytest.approx(35 / 60)


# ---------------------------------------------------------------------------
# eval_occasion_text_xval / majority baseline
# ---------------------------------------------------------------------------


def test_eval_occasion_text_xval_membership_not_exact_match() -> None:
    items = _items_df(
        [
            # tag_occasions finds {"casual", "party"} via "casual" and "party" keywords;
            # predicted "party" is a MEMBER even though it isn't the only tag found.
            {"article_id": 1, "title": "casual party top", "description": ""},
            {"article_id": 2, "title": "formal office wear", "description": ""},  # {"formal"}
            {"article_id": 3, "title": "a plain t-shirt", "description": ""},  # no signal
        ]
    )
    attr_index = {
        "1": {"occasion": "party", "occasion_confidence": 0.8},
        "2": {"occasion": "casual", "occasion_confidence": 0.2},  # wrong vs {"formal"}
        "3": {"occasion": "casual", "occasion_confidence": 0.5},
    }
    result = eval_occasion_text_xval(items, attr_index)

    assert result.n_total == 3
    assert result.n_covered == 2  # items 1, 2 (item 3 has no lexicon signal)
    assert result.n_evaluated == 2
    assert result.n_correct == 1  # item 1 only
    assert result.correct_confidences == [0.8]
    assert result.incorrect_confidences == [0.2]


def test_pool_occasion_results_sums_raw_counts() -> None:
    a = OccasionEvalResult(n_total=10, n_covered=5, n_evaluated=5, n_correct=4)
    b = OccasionEvalResult(n_total=5, n_covered=5, n_evaluated=5, n_correct=1)
    pooled = pool_occasion_results([a, b])

    assert pooled.n_total == 15
    assert pooled.n_covered == 10
    assert pooled.n_evaluated == 10
    assert pooled.n_correct == 5
    assert pooled.accuracy == pytest.approx(0.5)


def test_majority_predicted_label_picks_most_frequent() -> None:
    attr_index = {
        "1": {"occasion": "casual"},
        "2": {"occasion": "casual"},
        "3": {"occasion": "party"},
    }
    label, frac = majority_predicted_label(attr_index, "occasion")
    assert label == "casual"
    assert frac == pytest.approx(2 / 3)


def test_majority_predicted_label_empty_index() -> None:
    assert majority_predicted_label({}, "occasion") == ("", 0.0)


def test_majority_baseline_accuracy_matches_evaluated_denominator() -> None:
    """The baseline denominator must equal n_evaluated from eval_occasion_text_xval on the
    same inputs -- otherwise the honesty comparison in the report would not be apples-to-apples."""
    items = _items_df(
        [
            {"article_id": 1, "title": "casual party top", "description": ""},
            {"article_id": 2, "title": "formal office wear", "description": ""},
            {"article_id": 3, "title": "a plain t-shirt", "description": ""},
        ]
    )
    attr_index = {
        "1": {"occasion": "party", "occasion_confidence": 0.8},
        "2": {"occasion": "casual", "occasion_confidence": 0.2},
    }
    xval = eval_occasion_text_xval(items, attr_index)
    base_correct, base_evaluated = majority_baseline_accuracy(items, attr_index, None, "casual")

    assert base_evaluated == xval.n_evaluated == 2
    # item 1 -> tags {"casual", "party"}, "casual" is a member -> correct.
    # item 2 -> tags {"formal"}, "casual" is not a member -> incorrect.
    assert base_correct == 1


# ---------------------------------------------------------------------------
# confidence_gap
# ---------------------------------------------------------------------------


def test_confidence_gap_computes_means() -> None:
    mean_c, mean_i = confidence_gap([0.8, 0.6], [0.2, 0.4])
    assert mean_c == pytest.approx(0.7)
    assert mean_i == pytest.approx(0.3)


def test_confidence_gap_handles_empty_lists() -> None:
    assert confidence_gap([], []) == (None, None)
    mean_c, mean_i = confidence_gap([0.5], [])
    assert mean_c == pytest.approx(0.5)
    assert mean_i is None


# ---------------------------------------------------------------------------
# build_spotcheck_sample
# ---------------------------------------------------------------------------


def test_build_spotcheck_sample_only_includes_items_with_verified_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "P1.jpg").write_bytes(b"fake-jpeg-bytes")
    # P2.jpg intentionally NOT created -> article_id 2 must be excluded.

    items = _items_df(
        [
            {
                "article_id": 1,
                "product_id": "P1",
                "title": "black shirt",
                "description": "",
                "category": "Shirts",
            },
            {
                "article_id": 2,
                "product_id": "P2",
                "title": "white shirt",
                "description": "",
                "category": "Shirts",
            },
        ]
    )
    attr_index = {
        "1": {
            "color": "black",
            "color_confidence": 0.9,
            "pattern": "solid",
            "pattern_confidence": 0.5,
            "fabric": "cotton",
            "fabric_confidence": 0.3,
            "occasion": "casual",
            "occasion_confidence": 0.4,
        },
        "2": {
            "color": "white",
            "color_confidence": 0.9,
            "pattern": "solid",
            "pattern_confidence": 0.5,
            "fabric": "cotton",
            "fabric_confidence": 0.3,
            "occasion": "casual",
            "occasion_confidence": 0.4,
        },
    }

    sample = build_spotcheck_sample("testbrand", items, attr_index, images_dir, n=5, seed=42)

    assert len(sample) == 1
    entry = sample[0]
    assert entry["article_id"] == 1
    assert entry["brand"] == "testbrand"
    assert Path(entry["image_path"]).exists()
    assert entry["attributes"]["color"] == {"label": "black", "confidence": 0.9}


def test_build_spotcheck_sample_empty_when_no_eligible_items(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    items = _items_df(
        [{"article_id": 1, "product_id": "P1", "title": "x", "description": "", "category": "A"}]
    )
    sample = build_spotcheck_sample("testbrand", items, {}, images_dir, n=5, seed=42)
    assert sample == []
