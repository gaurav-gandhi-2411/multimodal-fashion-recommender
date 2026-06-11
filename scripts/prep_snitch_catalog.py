"""
Build an expanded Snitch catalog CSV from the full agentic-shopping-assistant raw data.

Output: data/snitch/catalog_full.csv  (CsvSource format, ready for ingest_catalog.py)

Strategy:
- Filter to apparel-only categories (no accessories/footwear)
- Deduplicate by title within each category (keep first occurrence = cheapest variant)
- Cap per-category at TARGET_PER_CAT; thin categories get all available items
- Construct pdp_url from handle column
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW = Path("C:/Users/gaura/ml-projects/agentic-shopping-assistant/data/raw/shopify/snitch/products.csv")
OUT = Path("data/snitch/catalog_full.csv")

# Apparel categories to include and per-category caps
CATEGORY_CAPS: dict[str, int] = {
    "Shirts": 200,
    "T-Shirts": 200,
    "Trousers": 200,
    "Jeans": 300,       # was 50 — key fix
    "Cargo Pants": 200,
    "Jackets": 200,     # was 50 — key fix
    "Overshirt": 200,
    "Sweaters": 150,
    "Shorts": 150,
    "Hoodies": 100,
}

df = pd.read_csv(RAW)
print(f"Raw rows: {len(df)}")

out_rows: list[dict] = []

for cat, cap in CATEGORY_CAPS.items():
    sub = df[df["type"] == cat].copy()
    # deduplicate by title, keep first (cheapest or earliest uploaded)
    sub = sub.drop_duplicates(subset=["title"], keep="first")
    sub = sub.head(cap)

    for _, row in sub.iterrows():
        pdp_url = f"https://snitch.co.in/products/{row['handle']}" if pd.notna(row.get("handle")) else ""
        # Append category suffix to title to match existing catalog convention
        title = f"{row['title']} ( {cat})"
        out_rows.append({
            "product_id": str(int(row["id"])),
            "title": title,
            "description": str(row.get("description", "")),
            "image_url": str(row["image_url"]),
            "price_inr": float(row["price_inr"]),
            "category": cat,
            "pdp_url": pdp_url,
        })

result = pd.DataFrame(out_rows)
print("\nCategory counts in output:")
print(result["category"].value_counts().to_string())
print(f"\nTotal rows: {len(result)}")

result.to_csv(OUT, index=False)
print(f"\nWritten to {OUT}")
