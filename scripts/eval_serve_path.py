"""scripts/eval_serve_path.py  -- Serve-path eval: local FAISS vs live HTTP API.

Purpose
-------
Answers the question a buyer's data scientist always asks first:
  "Can I reproduce your recall numbers by calling your API?"

Compares two code paths side-by-side for the same item set:
  --local      direct FAISS index search (no HTTP) — the training-time path
  --http-mode  calls the live API endpoints       — the production serve path

If local ≈ http the numbers in the README are reproducible via the API.
Any gap indicates that the serve path diverges from what was measured offline
(reranking, rate-limiting, indexing difference, etc.).

Modes
-----
style   (default)  POST /v1/{brand}/style-search?text=<title>
                   Metric: category recall@k — does the top-k contain any item
                   in the same category as the query?

visual             POST /v1/{brand}/visual-search  (multipart image upload)
                   Downloads catalog images from CDN; --sample limits the set.
                   Same category-recall metric.

Usage
-----
  # Local only (fast, no network needed):
  python scripts/eval_serve_path.py --mode style --local

  # HTTP only (proves API numbers):
  python scripts/eval_serve_path.py --mode style --http-mode \\
      --api-base https://fashion-recommender-staging-657468372797.asia-south1.run.app \\
      --api-key snitch-staging-key

  # Side-by-side comparison (default when both flags given):
  python scripts/eval_serve_path.py --mode style --local --http-mode \\
      --api-base https://... --api-key snitch-staging-key

  # Visual search (downloads images, capped at --sample items):
  python scripts/eval_serve_path.py --mode visual --local --http-mode \\
      --api-base https://... --api-key snitch-staging-key --sample 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Category-recall metric
# ---------------------------------------------------------------------------

def category_recall_at_k(
    results: list[dict],  # list of {"item_id": str, "category": str}
    query_category: str,
    k: int,
) -> bool:
    """True if any of the top-k results shares the query item's category."""
    for r in results[:k]:
        if r.get("category", "") == query_category:
            return True
    return False


# ---------------------------------------------------------------------------
# Local (direct FAISS) evaluation
# ---------------------------------------------------------------------------

def run_local_style(
    catalog,  # pandas DataFrame — full indexed catalog (for aid→category lookup)
    retriever,
    k: int,
    eval_rows: list,  # pre-sampled rows to evaluate
) -> dict:
    from app.visual import encode_query_text, get_image_encoder
    from tqdm import tqdm

    get_image_encoder()

    aid_to_cat = {int(r.article_id): str(getattr(r, "category", "") or "") for r in catalog.itertuples()}
    rows = eval_rows

    hits = 0
    total = 0
    for row in tqdm(rows, desc="local style", leave=False):
        title = str(getattr(row, "title", "") or "")
        if not title.strip():
            continue
        query_cat = str(getattr(row, "category", "") or "")
        emb = encode_query_text(title)
        results_raw = retriever.search(emb, k)
        retrieved = [
            {"item_id": str(aid), "category": aid_to_cat.get(int(aid), "")}
            for aid, _ in results_raw
        ]
        if category_recall_at_k(retrieved, query_cat, k):
            hits += 1
        total += 1

    return {"hits": hits, "total": total, "recall": hits / total if total else 0.0}


def run_local_visual(
    catalog,
    visual_retriever,
    k: int,
    eval_rows: list,
    image_cache_dir: Path,
) -> dict:
    import requests
    from app.visual import get_image_encoder
    from src.encoders.image_encoder import ImageEncoder
    from tqdm import tqdm
    import yaml

    get_image_encoder()
    cfg_path = Path("config.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg = {**cfg, "encoders": {**cfg["encoders"], "device": "cpu"}}
    enc = ImageEncoder(cfg)

    aid_to_cat = {int(r.article_id): str(getattr(r, "category", "") or "") for r in catalog.itertuples()}
    rows = eval_rows

    hits = 0
    total = 0
    for row in tqdm(rows, desc="local visual", leave=False):
        aid = int(row.article_id)
        image_url = str(getattr(row, "image_url", "") or "")
        query_cat = str(getattr(row, "category", "") or "")
        if not image_url:
            continue

        cache_path = image_cache_dir / f"{aid}.jpg"
        if not cache_path.exists():
            try:
                r = requests.get(image_url, timeout=10)
                r.raise_for_status()
                cache_path.write_bytes(r.content)
            except Exception:
                continue
        try:
            embs = enc.encode_batch([cache_path])
        except Exception:
            continue

        emb = embs[0]
        results_raw = visual_retriever.search(emb, k)
        retrieved = [
            {"item_id": str(r_aid), "category": aid_to_cat.get(int(r_aid), "")}
            for r_aid, _ in results_raw
        ]
        if category_recall_at_k(retrieved, query_cat, k):
            hits += 1
        total += 1

    return {"hits": hits, "total": total, "recall": hits / total if total else 0.0}


# ---------------------------------------------------------------------------
# HTTP evaluation
# ---------------------------------------------------------------------------

def run_http_style(
    eval_rows: list,
    aid_to_cat: dict,
    api_base: str,
    api_key: str,
    brand: str,
    k: int,
    delay: float = 1.1,
) -> dict:
    import requests
    from tqdm import tqdm

    rows = eval_rows
    hits = 0
    total = 0
    errors = 0

    url = f"{api_base.rstrip('/')}/v1/{brand}/style-search"
    for row in tqdm(rows, desc="http style", leave=False):
        title = str(getattr(row, "title", "") or "")
        query_cat = str(getattr(row, "category", "") or "")
        if not title.strip():
            continue
        try:
            for attempt in range(3):
                resp = requests.post(
                    url,
                    params={"text": title, "k": k},
                    headers={"X-Api-Key": api_key, "Content-Length": "0"},
                    timeout=15,
                )
                if resp.status_code == 429 and attempt < 2:
                    time.sleep(delay * 2 ** attempt)
                    continue
                resp.raise_for_status()
                break
            data = resp.json()
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"\n  HTTP error: {exc}")
            continue

        results_raw = data.get("results", [])
        # Enrich result categories from local catalog (API returns item_id + score)
        retrieved = []
        for r in results_raw:
            iid = r.get("item_id", "")
            try:
                cat = aid_to_cat.get(int(iid), "")
            except (ValueError, TypeError):
                cat = ""
            retrieved.append({"item_id": iid, "category": cat})

        if category_recall_at_k(retrieved, query_cat, k):
            hits += 1
        total += 1
        time.sleep(delay)

    return {"hits": hits, "total": total, "recall": hits / total if total else 0.0, "errors": errors}


def run_http_visual(
    eval_rows: list,
    aid_to_cat: dict,
    api_base: str,
    api_key: str,
    brand: str,
    k: int,
    image_cache_dir: Path,
    delay: float = 1.1,
) -> dict:
    import requests
    from tqdm import tqdm

    rows = eval_rows
    hits = 0
    total = 0
    errors = 0

    url = f"{api_base.rstrip('/')}/v1/{brand}/visual-search"
    for row in tqdm(rows, desc="http visual", leave=False):
        aid = int(row.article_id)
        image_url = str(getattr(row, "image_url", "") or "")
        query_cat = str(getattr(row, "category", "") or "")
        if not image_url:
            continue

        cache_path = image_cache_dir / f"{aid}.jpg"
        if not cache_path.exists():
            try:
                r = requests.get(image_url, timeout=10)
                r.raise_for_status()
                cache_path.write_bytes(r.content)
            except Exception:
                errors += 1
                continue

        try:
            with open(cache_path, "rb") as fh:
                resp = requests.post(
                    url,
                    files={"image": (f"{aid}.jpg", fh, "image/jpeg")},
                    headers={"X-Api-Key": api_key},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"\n  HTTP error: {exc}")
            continue

        results_raw = data.get("results", [])
        retrieved = []
        for r in results_raw:
            iid = r.get("item_id", "")
            try:
                cat = aid_to_cat.get(int(iid), "")
            except (ValueError, TypeError):
                cat = ""
            retrieved.append({"item_id": iid, "category": cat})

        if category_recall_at_k(retrieved, query_cat, k):
            hits += 1
        total += 1
        time.sleep(delay)

    return {"hits": hits, "total": total, "recall": hits / total if total else 0.0, "errors": errors}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["style", "visual"], default="style")
    parser.add_argument("--local", action="store_true", help="Run local FAISS evaluation")
    parser.add_argument("--http-mode", action="store_true", dest="http_mode",
                        help="Run HTTP API evaluation")
    parser.add_argument("--brand", default="snitch")
    parser.add_argument("--k", type=int, default=5, help="Recall@k (default 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N items (default: all for style, 200 for visual)")
    parser.add_argument("--api-base", default="https://fashion-recommender-staging-657468372797.asia-south1.run.app")
    parser.add_argument("--api-key", default="snitch-staging-key")
    parser.add_argument("--catalog", default=None, help="Catalog parquet (default: data/{brand}/items.parquet)")
    parser.add_argument("--index-dir", default=None, help="Visual index dir (default: indices/{brand}/visual.faiss)")
    parser.add_argument("--delay", type=float, default=1.1,
                        help="Seconds between HTTP requests (default 1.1 — respects 60/min rate limit)")
    args = parser.parse_args()

    # Default sample sizes
    if args.sample is None and args.mode == "visual":
        args.sample = 200

    run_local = args.local
    run_http = args.http_mode
    if not run_local and not run_http:
        # Default: run both when an API base is set and a local index exists
        run_local = True
        run_http = True

    import pandas as pd
    from src.retrieval.faiss_index import FaissRetriever

    catalog_path = args.catalog or f"data/{args.brand}/items.parquet"
    index_dir = args.index_dir or f"indices/{args.brand}/visual.faiss"

    catalog = pd.read_parquet(catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)

    image_cache_dir = Path(f".image_cache/{args.brand}")
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Brand:     {args.brand}")
    print(f"Mode:      {args.mode}")
    print(f"Recall@k:  k={args.k}")
    print(f"Catalog:   {len(catalog)} items  (sample={args.sample or 'all'})")
    print(f"Paths:     catalog={catalog_path}  index={index_dir}")
    print()

    local_result: dict | None = None
    http_result: dict | None = None

    # Always load the FAISS index so both paths query the same indexed item set.
    visual_retriever = FaissRetriever.load(index_dir)
    index_ids = {int(x) for x in visual_retriever.article_ids}
    catalog_indexed = catalog[catalog["article_id"].isin(index_ids)].reset_index(drop=True)
    print(f"Index:  {len(catalog_indexed)}/{len(catalog)} items in visual index")

    # Sample once so local and HTTP eval compare identical items.
    import random
    rows_all = list(catalog_indexed.itertuples())
    if args.sample:
        random.seed(42)
        eval_rows = random.sample(rows_all, min(args.sample, len(rows_all)))
    else:
        eval_rows = rows_all
    print(f"Sample: {len(eval_rows)} items (seed=42)")
    print()

    if run_local:
        if args.mode == "style":
            local_result = run_local_style(catalog_indexed, visual_retriever, args.k, eval_rows)
        else:
            local_result = run_local_visual(catalog_indexed, visual_retriever, args.k, eval_rows, image_cache_dir)

    if run_http:
        print(f"HTTP:   {args.api_base}")
        aid_to_cat = {int(r.article_id): str(getattr(r, "category", "") or "") for r in catalog_indexed.itertuples()}
        if args.mode == "style":
            http_result = run_http_style(eval_rows, aid_to_cat, args.api_base, args.api_key, args.brand, args.k, args.delay)
        else:
            http_result = run_http_visual(eval_rows, aid_to_cat, args.api_base, args.api_key, args.brand, args.k, image_cache_dir, args.delay)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    w = 60
    print()
    print("=" * w)
    print(f"SERVE-PATH EVAL: {args.mode.upper()} SEARCH  (category recall@{args.k})")
    print("=" * w)

    if local_result:
        r = local_result
        print(f"  Local  (direct FAISS):  {r['recall']:.4f}  ({r['hits']}/{r['total']})")

    if http_result:
        r = http_result
        errs = r.get("errors", 0)
        err_str = f"  [{errs} HTTP errors]" if errs else ""
        print(f"  HTTP   (live API):      {r['recall']:.4f}  ({r['hits']}/{r['total']}){err_str}")

    if local_result and http_result:
        gap = abs(local_result["recall"] - http_result["recall"])
        pct = gap / max(local_result["recall"], 1e-9) * 100
        verdict = "PASS (gap < 2%)" if pct < 2.0 else f"REVIEW (gap {pct:.1f}%)"
        print()
        print(f"  Gap:   {gap:.4f}  ({pct:.1f}%)  ->  {verdict}")
        print()
        print("  A gap < 2% confirms the API numbers reproduce the local eval.")
        print("  Larger gaps indicate serve-path differences (reranking, indexing).")

    print("=" * w)


if __name__ == "__main__":
    main()
