"""tests/test_eval_honest_style_queries.py -- unit tests for the honest free-text
style-search eval fixture loader (no model, no FAISS, no network I/O)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_honest_style_queries import FIXTURES_DIR, load_honest_queries  # noqa: E402

_BRANDS = ["snitch", "fashor", "powerlook"]


def test_fixture_files_exist_for_all_three_brands():
    for brand in _BRANDS:
        assert (FIXTURES_DIR / f"{brand}.json").exists()


def test_fixture_has_at_least_30_queries_per_brand():
    for brand in _BRANDS:
        rows = load_honest_queries(brand)
        assert len(rows) >= 30, f"{brand} has only {len(rows)} queries"


def test_loaded_rows_expose_title_and_category_attributes():
    rows = load_honest_queries("snitch")
    assert all(hasattr(r, "title") and hasattr(r, "category") for r in rows)
    assert all(isinstance(r.title, str) and r.title.strip() for r in rows)
    assert all(isinstance(r.category, str) and r.category.strip() for r in rows)


def test_queries_are_not_verbatim_category_names():
    """Each query must be a realistic free-text phrase, not just the bare category label
    -- otherwise this eval would collapse back into the same lexical-match issue it's
    meant to fix."""
    for brand in _BRANDS:
        rows = load_honest_queries(brand)
        for r in rows:
            assert r.title.strip().lower() != r.category.strip().lower()
            assert len(r.title.split()) >= 3, f"query too short: {r.title!r}"


def test_expected_categories_are_real_categories_in_the_catalog():
    """Every fixture's expected_category must exist in that brand's real catalog taxonomy
    -- otherwise the eval would silently measure recall against a category that can never
    be retrieved."""
    import pandas as pd

    for brand in _BRANDS:
        catalog_path = REPO_ROOT / "data" / brand / "items.parquet"
        if not catalog_path.exists():
            continue  # local data not present in this environment -- skip, not a failure
        real_categories = set(pd.read_parquet(catalog_path)["category"].unique())
        rows = load_honest_queries(brand)
        for r in rows:
            assert r.category in real_categories, (
                f"{brand}: expected_category {r.category!r} not found in real catalog "
                f"categories {sorted(real_categories)}"
            )


def test_fixture_json_is_well_formed():
    for brand in _BRANDS:
        path = FIXTURES_DIR / f"{brand}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        for entry in data:
            assert set(entry.keys()) == {"query", "expected_category"}
