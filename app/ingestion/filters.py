"""Generic post-fetch catalog filters, applied after a CatalogSource returns rows and
before the ingestion pipeline downloads images or computes embeddings.
"""
from __future__ import annotations

from collections import Counter

from app.ingestion.schema import CatalogRow


def filter_excluded_categories(
    rows: list[CatalogRow], excluded_categories: set[str]
) -> tuple[list[CatalogRow], list[CatalogRow]]:
    """Split catalog rows into (kept, excluded) by case-insensitive exact category match.

    Some Shopify catalogs mix non-apparel SKUs (fragrances, gift cards) into an otherwise
    apparel catalog. A CLIP visual embedding of a perfume bottle or a gift-card graphic has
    no meaningful similarity relationship to garments, so leaving them in the visual index
    surfaces as a "similar item" to a dress query. This is a property of the CATEGORY, not
    any one brand, so the filter is a generic, opt-in, per-invocation exclusion list rather
    than a hardcoded brand-specific rule or a silent built-in default.
    """
    if not excluded_categories:
        return rows, []
    normalized = {c.strip().lower() for c in excluded_categories}
    kept: list[CatalogRow] = []
    excluded: list[CatalogRow] = []
    for row in rows:
        if row.category.strip().lower() in normalized:
            excluded.append(row)
        else:
            kept.append(row)
    return kept, excluded


def excluded_category_breakdown(excluded: list[CatalogRow]) -> Counter[str]:
    """Category -> count, for reporting what a --exclude-categories run actually dropped."""
    return Counter(row.category for row in excluded)
