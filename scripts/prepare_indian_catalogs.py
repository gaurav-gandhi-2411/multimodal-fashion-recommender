"""
Prepare Indian brand catalogs (Snitch, Fashor, Powerlook) for ingestion.

Reads parquet files from the sibling agentic-shopping-assistant project,
transforms them into the 7-column CSV format required by scripts/ingest_catalog.py,
and writes them to data/{brand}/catalog.csv.

Sampling strategy:
- Powerlook : ALL rows  (922 items — small enough to ingest fully)
- Fashor    : ALL rows  (3618 items — ethnic vocabulary coverage needed)
- Snitch    : Stratified 500-item sample from top-10 categories, min(count, 50) per
              category, seed=42 for reproducibility.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIBLING_ROOT = Path(r"C:\Users\gaura\ml-projects\agentic-shopping-assistant")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARQUET_SOURCES: dict[str, Path] = {
    "snitch": SIBLING_ROOT / "data" / "processed" / "snitch" / "catalogue.parquet",
    "fashor": SIBLING_ROOT / "data" / "processed" / "fashor" / "catalogue.parquet",
    "powerlook": SIBLING_ROOT / "data" / "processed" / "powerlook" / "catalogue.parquet",
}

PDP_BASES: dict[str, str] = {
    "snitch": "https://snitch.co.in/products/",
    "fashor": "https://fashor.com/products/",
    "powerlook": "https://powerlook.in/products/",
}

TARGET_COLUMNS = [
    "product_id", "title", "description", "image_url", "price_inr", "category", "pdp_url"
]

SNITCH_SAMPLE_SIZE = 500
SNITCH_PER_CATEGORY = 50
SNITCH_TOP_N_CATS = 10
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_category(raw: str) -> str:
    """Strip trailing `]]=`, `]`, `=`, and whitespace from category names.

    Example: "3P Kurta Set]]=  " -> "3P Kurta Set"
    """
    return re.sub(r"[\]\[=\s]+$", "", raw).strip()


def build_df(brand: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Transform raw parquet columns into the 7-column catalog schema.

    Drops rows where:
    - detail_desc is null/empty after strip
    - image_url is null/empty
    - price_inr <= 0 or null
    - pdp_handle is null/empty
    """
    df = raw.copy()

    # product_id
    df["product_id"] = df["article_id"].astype(str).str.strip()

    # title: display_name if non-empty, else prod_name
    display = df["display_name"].fillna("").str.strip()
    prod = df["prod_name"].fillna("").str.strip()
    df["title"] = display.where(display != "", prod)

    # description
    df["description"] = df["detail_desc"].fillna("").str.strip()
    df = df[df["description"] != ""]

    # image_url
    df["image_url"] = df["image_url"].fillna("").str.strip()
    df = df[df["image_url"] != ""]

    # price_inr
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df = df[df["price_inr"].notna() & (df["price_inr"] > 0)]

    # category
    df["category"] = df["product_type_name"].fillna("").str.strip().apply(clean_category)

    # pdp_url
    pdp_base = PDP_BASES[brand]
    df["pdp_handle_clean"] = df["pdp_handle"].fillna("").str.strip()
    df = df[df["pdp_handle_clean"] != ""]
    df["pdp_url"] = pdp_base + df["pdp_handle_clean"]

    return df[TARGET_COLUMNS]


def stratified_snitch(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stratified 500-item sample from the top-10 Snitch categories.

    For each of the top 10 categories by item count, samples min(count, 50) rows.
    Seed=42 for reproducibility.
    """
    top_cats = (
        df["category"]
        .value_counts()
        .head(SNITCH_TOP_N_CATS)
        .index.tolist()
    )
    parts: list[pd.DataFrame] = []
    for cat in top_cats:
        cat_df = df[df["category"] == cat]
        n = min(len(cat_df), SNITCH_PER_CATEGORY)
        parts.append(cat_df.sample(n=n, random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def print_summary(brand: str, df: pd.DataFrame) -> None:
    """Print item count, category breakdown, and 3 sample PDP URLs."""
    print(f"\n{'=' * 60}")
    print(f"Brand: {brand.upper()}  ({len(df)} items)")
    print("Categories:")
    cat_counts = df["category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt}")
    print("Sample PDP URLs:")
    for url in df["pdp_url"].head(3).tolist():
        print(f"  {url}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_brand(brand: str) -> None:
    """Load, transform, (optionally sample), and write catalog CSV for one brand."""
    src = PARQUET_SOURCES[brand]
    raw = pd.read_parquet(src)
    df = build_df(brand, raw)

    if brand == "snitch":
        df = stratified_snitch(df)

    out_dir = PROJECT_ROOT / "data" / brand
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "catalog.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print_summary(brand, df)
    print(f"Written: {out_path}")


def main() -> None:
    """Process all three Indian brand catalogs."""
    for brand in ["snitch", "fashor", "powerlook"]:
        process_brand(brand)

    print("\nAll catalogs prepared successfully.")


if __name__ == "__main__":
    main()
