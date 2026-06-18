"""Update all snitch.co.in URLs to snitch.com in catalog files."""
from __future__ import annotations

import json
import re

import pandas as pd

BASE = "C:/Users/gaura/ml-projects/multimodal-fashion-recommender"

OLD = "snitch.co.in"
NEW = "snitch.com"

# 1. Update items.parquet
parquet_path = f"{BASE}/data/snitch/items.parquet"
df = pd.read_parquet(parquet_path)
before = df["pdp_url"].str.contains(OLD, na=False).sum()
df["pdp_url"] = df["pdp_url"].str.replace(OLD, NEW, regex=False)
after_co_in = df["pdp_url"].str.contains(OLD, na=False).sum()
after_com = df["pdp_url"].str.contains(NEW, na=False).sum()
df.to_parquet(parquet_path, index=False)
print(f"parquet: {before} co.in URLs -> {after_com} snitch.com URLs (remaining co.in: {after_co_in})")

# 2. Update catalog.csv
csv_path = f"{BASE}/data/snitch/catalog.csv"
with open(csv_path, encoding="utf-8") as f:
    content = f.read()
n = content.count(OLD)
content = content.replace(OLD, NEW)
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"catalog.csv: replaced {n} occurrences")

# 3. Update demo catalog JSON
json_path = f"{BASE}/demo/public/catalog/snitch.json"
with open(json_path, encoding="utf-8") as f:
    catalog = json.load(f)
updated = 0
for item in catalog.values():
    if OLD in item.get("pdp_url", ""):
        item["pdp_url"] = item["pdp_url"].replace(OLD, NEW)
        updated += 1
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, separators=(",", ":"))
print(f"catalog.json: updated {updated} pdp_urls")
