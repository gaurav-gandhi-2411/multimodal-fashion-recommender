"""scripts/eval_style_search.py -- Serve-path recall@5 eval for /v1/{brand}/style-search.

Runs the EXACT production code path (encode_query_text -> visual FAISS search)
using the live Snitch visual index, not a mock.

Metric: recall@5 = fraction of items whose top-5 text-query results include the
        target item itself (self-retrieval using prod_name as the text query).

This is the minimal bar for CLIP text-image alignment: if a product's own name
cannot retrieve the product in top-5, the style-search feature is not useful.

Usage:
    python scripts/eval_style_search.py
    python scripts/eval_style_search.py --brand fashor --k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="snitch")
    parser.add_argument("--k", type=int, default=5, help="Recall@k")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Visual index directory (default: indices/{brand}/visual.faiss)",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Catalog parquet (default: data/{brand}/items.parquet)",
    )
    args = parser.parse_args()

    brand = args.brand
    k = args.k
    index_dir = args.index_dir or f"indices/{brand}/visual.faiss"
    catalog_path = args.catalog or f"data/{brand}/items.parquet"

    print(f"Brand:        {brand}")
    print(f"Visual index: {index_dir}")
    print(f"Catalog:      {catalog_path}")
    print(f"Recall@{k} eval\n")

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from app.visual import encode_query_text, get_image_encoder
    from src.retrieval.faiss_index import FaissRetriever

    # Warm up the CLIP singleton once (loads model ~1s)
    print("Loading CLIP text encoder... ", end="", flush=True)
    get_image_encoder()
    print("done")

    print("Loading visual FAISS index... ", end="", flush=True)
    retriever = FaissRetriever.load(index_dir)
    index_ids: set[int] = set(int(x) for x in retriever.article_ids)
    aid_to_row: dict[int, int] = {int(aid): i for i, aid in enumerate(retriever.article_ids)}
    print(f"done ({retriever.index.ntotal} items, d={retriever.index.d})")

    catalog = pd.read_parquet(catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)

    # Keep only items present in the visual index
    catalog = catalog[catalog["article_id"].isin(index_ids)].reset_index(drop=True)
    print(f"Catalog items in visual index: {len(catalog)}\n")

    hits_self = 0      # exact item in top-k
    hits_cat = 0       # any item from same category in top-k
    total = 0
    low_conf: list[tuple[float, str]] = []  # (confidence, query) for bottom-10

    # Build aid→category lookup for category-level recall
    aid_to_cat: dict[int, str] = {
        int(r["article_id"]): str(r.get("category", ""))
        for _, r in catalog.iterrows()
    }

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc=f"Eval recall@{k}"):
        article_id = int(row["article_id"])
        title = str(row.get("title", row.get("prod_name", "")))
        target_cat = str(row.get("category", ""))

        if not title.strip():
            continue

        query_emb = encode_query_text(title)
        results = retriever.search(query_emb, k)

        # C3: match_confidence (same formula as API)
        if results:
            top_score = float(results[0][1])
            pool = [float(s) for _, s in results[:k]]
            conf = round(top_score - min(pool), 4) if len(pool) > 1 else 0.0
        else:
            conf = 0.0

        retrieved_ids = [int(r[0]) for r in results]
        retrieved_cats = [aid_to_cat.get(rid, "") for rid in retrieved_ids]

        hit_self = article_id in set(retrieved_ids)
        hit_cat = target_cat in retrieved_cats if target_cat else False

        if hit_self:
            hits_self += 1
        if hit_cat:
            hits_cat += 1
        if not hit_self:
            low_conf.append((conf, title))

        total += 1

    recall_self = hits_self / total if total > 0 else 0.0
    recall_cat = hits_cat / total if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"STYLE-SEARCH EVAL (k={k})")
    print(f"{'=' * 60}")
    print(f"  Items evaluated:           {total}")
    print()
    print(f"  Category recall@{k}:         {recall_cat:.4f}  ({recall_cat * 100:.1f}%)")
    print(f"  (bar >=0.90):              {'PASS' if recall_cat >= 0.90 else 'FAIL'}")
    print()
    print(f"  Self-retrieval recall@{k}:   {recall_self:.4f}  ({recall_self * 100:.1f}%)")
    print(f"  (secondary; CLIP optimises semantic similarity, not identity lookup)")
    print()

    # Show 10 worst misses by query text
    low_conf.sort(key=lambda x: x[0])
    print("Bottom-10 self-retrieval misses (lowest match_confidence):")
    for conf, title in low_conf[:10]:
        print(f"  conf={conf:.4f}  query={title[:80]!r}")

    print()

    # Spot-check: a few example queries
    examples = [
        "white oversized cotton shirt",
        "cobalt blue linen shirt under 1200",
        "black slim fit chinos",
        "floral print casual shirt",
        "navy blue formal trousers",
    ]
    print("Spot-check queries (top-3 results):")
    for query in examples:
        emb = encode_query_text(query)
        hits = retriever.search(emb, 3)
        pool = [float(s) for _, s in hits[:k]]
        conf = round(float(hits[0][1]) - min(pool), 4) if len(pool) > 1 else 0.0
        top_ids = [int(h[0]) for h in hits]
        top_titles = []
        for tid in top_ids:
            row_meta = catalog[catalog["article_id"] == tid]
            if not row_meta.empty:
                t = str(row_meta.iloc[0].get("title", row_meta.iloc[0].get("prod_name", "")))
                top_titles.append(t[:50])
            else:
                top_titles.append(f"id={tid}")
        print(f"\n  Query: {query!r}  (confidence={conf:.4f})")
        for i, (tid, ttl) in enumerate(zip(top_ids, top_titles), 1):
            print(f"    {i}. [{tid}] {ttl}")


if __name__ == "__main__":
    main()
