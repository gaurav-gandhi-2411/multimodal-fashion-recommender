"""Per-brand category-string canonicalization.

Some catalogs' raw `product_type` field (from Shopify or CSV export) contains
near-duplicate labels for the same real category -- typically a singular/plural spelling
split from the source feed. This splits one real category into two labels for rerank
category-affinity matching and any category-match eval, understating both.

This module fixes the STRING, not the model or embeddings. Each entry below was confirmed
by comparing item counts and titles, not merged automatically by string similarity --
confirmed distinct real categories (e.g. Fashor's "Kurta Set", a multi-piece set, vs
"Kurta"/"Kurtas", a single garment) are deliberately NOT merged even though they share a
root word.
"""
from __future__ import annotations

# Human-reviewed, one-time-verified mappings. Canonical label chosen as the majority
# spelling in each brand's own catalog (the minority variant folds into the majority, not
# the other way around).
CATEGORY_CANONICALIZATION: dict[str, dict[str, str]] = {
    "fashor": {
        "Dress": "Dresses",  # 17 vs 192 items -- singular/plural split of the same category
        "Kurta": "Kurtas",  # 56 vs 462 items -- singular/plural split of the same category
        # "Kurta Set" (62 items) is a genuinely different product (a multi-piece set, not
        # a single garment) -- deliberately NOT merged into Kurta/Kurtas.
    },
    "virgio": {
        "Skort": "Skorts",  # 1 vs 3 items -- singular/plural split of the same category
    },
}


def canonicalize_category(brand: str, category: str) -> str:
    """Return the canonical category label for *category* in *brand*'s catalog.

    No-op (returns *category* unchanged) for brands or categories with no registered
    canonicalization -- this is an opt-in, per-brand, human-reviewed correction, not a
    generic fuzzy-matching pass.
    """
    return CATEGORY_CANONICALIZATION.get(brand, {}).get(category, category)
