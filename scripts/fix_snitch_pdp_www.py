"""Add the missing `www.` to all snitch.com PDP URLs.

Root cause: scripts/fix_snitch_domain.py (the co.in -> com migration, #16) did a bare
string replace and never added `www.`. Unlike snitch.co.in, snitch.com only redirects
bare-domain -> www at the root; /products/<slug> without www returns 404. Verified live
on 2026-07-07: 5/5 sampled bare-domain PDP URLs 404, 5/5 www URLs 200 (1803/1803 rows
affected). See scripts/prepare_indian_catalogs.py PDP_BASES for the source-of-truth fix.
"""
from __future__ import annotations

import json

import pandas as pd

BASE = "C:/Users/gaura/ml-projects/multimodal-fashion-recommender"

OLD = "https://snitch.com/"
NEW = "https://www.snitch.com/"


def fix_parquet(path: str) -> None:
    df = pd.read_parquet(path)
    before = df["pdp_url"].str.startswith(OLD, na=False).sum()
    df["pdp_url"] = df["pdp_url"].str.replace(OLD, NEW, regex=False)
    after = df["pdp_url"].str.startswith(NEW, na=False).sum()
    df.to_parquet(path, index=False)
    print(f"{path}: {before} bare-domain URLs -> {after} www URLs")


def fix_csv(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    n = content.count(OLD)
    content = content.replace(OLD, NEW)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"{path}: replaced {n} occurrences")


def fix_json(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    updated = 0
    for item in catalog.values():
        if item.get("pdp_url", "").startswith(OLD):
            item["pdp_url"] = NEW + item["pdp_url"][len(OLD) :]
            updated += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"))
    print(f"{path}: updated {updated} pdp_urls")


if __name__ == "__main__":
    fix_parquet(f"{BASE}/data/snitch/items.parquet")
    fix_csv(f"{BASE}/data/snitch/catalog.csv")
    fix_csv(f"{BASE}/data/snitch/catalog_full.csv")
    fix_json(f"{BASE}/demo/public/catalog/snitch.json")
