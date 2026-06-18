from __future__ import annotations

import json
import os

import pandas as pd

BASE = "C:/Users/gaura/ml-projects/multimodal-fashion-recommender"
OUT = f"{BASE}/demo/public/catalog"
os.makedirs(OUT, exist_ok=True)

for brand in ("snitch", "fashor", "powerlook"):
    df = pd.read_parquet(f"{BASE}/data/{brand}/items.parquet")
    catalog: dict[str, dict] = {}
    for _, row in df.iterrows():
        catalog[str(int(row["article_id"]))] = {
            "title": row["title"],
            "image_url": row["image_url"],
            "price_inr": int(row["price_inr"]) if pd.notna(row["price_inr"]) else 0,
            "pdp_url": row.get("pdp_url", "") or "",
            "category": row.get("category", "") or "",
        }
    out_path = f"{OUT}/{brand}.json"
    with open(out_path, "w") as f:
        json.dump(catalog, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) // 1024
    print(f"{brand}: {len(catalog)} items saved to {out_path} ({size_kb} KB)")
