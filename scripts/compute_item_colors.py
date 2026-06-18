"""Pre-compute dominant RGB/HSV colors for catalog items from thumbnail images.

Outputs demo/public/catalog/{brand}_colors.json:
  { "<item_id>": { "h": 210.5, "s": 0.45, "v": 0.88, "hex": "#3a7fd5" }, ... }

Uses Shopify CDN width parameter to download 48×48 thumbnails (~3-5KB each)
instead of full images, keeping total bandwidth under 30MB for all 3 brands.
"""
from __future__ import annotations

import colorsys
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

BASE = Path("C:/Users/gaura/ml-projects/multimodal-fashion-recommender")
BRANDS = ["snitch", "fashor", "powerlook"]
THUMB_SIZE = 48  # px — small enough for fast downloads, large enough for color accuracy
MAX_WORKERS = 80
REQUEST_TIMEOUT = 10  # seconds per image


def make_thumb_url(url: str) -> str:
    """Append width parameter for Shopify CDN images."""
    if "cdn.shopify.com" in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}width={THUMB_SIZE}"
    return url


def extract_color(img_bytes: bytes) -> dict[str, float | str]:
    """Average all pixels in a thumbnail to get dominant color impression."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((THUMB_SIZE, THUMB_SIZE))
    pixels = list(img.getdata())
    n = len(pixels)
    r = sum(p[0] for p in pixels) / n
    g = sum(p[1] for p in pixels) / n
    b = sum(p[2] for p in pixels) / n
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return {
        "h": round(h * 360, 1),
        "s": round(s, 3),
        "v": round(v, 3),
        "hex": "#{:02x}{:02x}{:02x}".format(int(round(r)), int(round(g)), int(round(b))),
    }


def fetch_color(item_id: str, url: str) -> tuple[str, dict | None]:
    try:
        resp = requests.get(make_thumb_url(url), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return item_id, extract_color(resp.content)
    except Exception:
        return item_id, None


def process_brand(brand: str) -> None:
    parquet_path = BASE / "data" / brand / "items.parquet"
    out_path = BASE / "demo" / "public" / "catalog" / f"{brand}_colors.json"

    df = pd.read_parquet(parquet_path, columns=["article_id", "image_url"])
    rows = [(str(row.article_id), row.image_url) for row in df.itertuples() if row.image_url]
    print(f"[{brand}] {len(rows)} items to process")

    colors: dict[str, dict] = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_color, iid, url): iid for iid, url in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            item_id, color = fut.result()
            if color:
                colors[item_id] = color
            else:
                failed += 1
            if done % 200 == 0 or done == len(rows):
                print(f"  [{brand}] {done}/{len(rows)} done ({failed} failed)")

    out_path.write_text(json.dumps(colors, separators=(",", ":")))
    print(f"[{brand}] saved {len(colors)} colors to {out_path.name} ({out_path.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    for brand in BRANDS:
        process_brand(brand)
    print("Done.")
