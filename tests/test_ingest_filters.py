from __future__ import annotations

from app.ingestion.filters import excluded_category_breakdown, filter_excluded_categories
from app.ingestion.schema import CatalogRow


def _row(product_id: str, category: str) -> CatalogRow:
    return CatalogRow(
        product_id=product_id,
        title=f"Item {product_id}",
        description="test description",
        image_url="https://cdn.example.com/img.jpg",
        price_inr=999.0,
        category=category,
        pdp_url="https://example.com/products/item",
    )


class TestFilterExcludedCategories:
    def test_empty_exclusion_set_returns_everything(self):
        rows = [_row("1", "Dress"), _row("2", "Fragrance")]
        kept, excluded = filter_excluded_categories(rows, set())
        assert kept == rows
        assert excluded == []

    def test_excludes_exact_category_match(self):
        rows = [_row("1", "Dress"), _row("2", "Fragrance"), _row("3", "Gift-Card")]
        kept, excluded = filter_excluded_categories(rows, {"Fragrance", "Gift-Card"})
        assert [r.product_id for r in kept] == ["1"]
        assert [r.product_id for r in excluded] == ["2", "3"]

    def test_case_insensitive_match(self):
        rows = [_row("1", "FRAGRANCE"), _row("2", "fragrance"), _row("3", "Dress")]
        kept, excluded = filter_excluded_categories(rows, {"fragrance"})
        assert [r.product_id for r in kept] == ["3"]
        assert len(excluded) == 2

    def test_partial_category_name_not_excluded(self):
        # "Dress" must not be excluded by an exclusion list targeting "Dresses" -- exact
        # match only, no substring/prefix matching that could silently over-exclude.
        rows = [_row("1", "Dress")]
        kept, excluded = filter_excluded_categories(rows, {"Dresses"})
        assert kept == rows
        assert excluded == []

    def test_excluding_every_row_returns_empty_kept_list(self):
        rows = [_row("1", "Fragrance"), _row("2", "Fragrance")]
        kept, excluded = filter_excluded_categories(rows, {"Fragrance"})
        assert kept == []
        assert len(excluded) == 2


class TestExcludedCategoryBreakdown:
    def test_counts_per_category(self):
        excluded = [_row("1", "Fragrance"), _row("2", "Fragrance"), _row("3", "Gift-Card")]
        breakdown = excluded_category_breakdown(excluded)
        assert breakdown["Fragrance"] == 2
        assert breakdown["Gift-Card"] == 1

    def test_empty_list_returns_empty_counter(self):
        assert excluded_category_breakdown([]) == {}
