"""scripts/diagnose_style_search_failures.py

Per-query diagnostic for the honest free-text style-search eval
(evals/fixtures/style_search_queries/{brand}.json). Prints, for every query: the expected
category, the rank-1 FAISS hit's category (what the rerank layer infers the query category
to be), and the final top-k categories actually returned -- so a MISS can be attributed to
either (a) FAISS retrieval itself missing the target category entirely, or (b) rank-1
landing on the wrong category and the rerank cascade following it.

Usage:
    python scripts/diagnose_style_search_failures.py --brand fashor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", default="fashor")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--only-misses", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    import pandas as pd
    import yaml
    from eval_honest_style_queries import load_honest_queries

    from app.brands.registry import BrandConfig
    from app.rerank import CategoryAffinityMap
    from app.rerank import rerank as _rerank
    from app.visual import encode_query_text, get_image_encoder
    from src.retrieval.faiss_index import FaissRetriever

    get_image_encoder()

    brand_yaml_path = REPO_ROOT / "brands" / f"{args.brand}.yaml"
    with brand_yaml_path.open() as f:
        brand_cfg = BrandConfig.model_validate(yaml.safe_load(f))
    rerank_cfg = brand_cfg.rerank

    catalog = pd.read_parquet(REPO_ROOT / "data" / args.brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)

    retriever = FaissRetriever.load(brand_cfg.visual_index_path)
    index_ids = {int(x) for x in retriever.article_ids}
    catalog_indexed = catalog[catalog["article_id"].isin(index_ids)].reset_index(drop=True)
    art_map = catalog_indexed.set_index("article_id").to_dict("index")
    aid_to_cat = {int(r.article_id): str(r.category or "") for r in catalog_indexed.itertuples()}

    pool_k = rerank_cfg.candidate_pool_size if rerank_cfg.enabled else args.k
    affinity_map = CategoryAffinityMap(rerank_cfg)

    honest_rows = load_honest_queries(args.brand)
    hits = 0
    affinity_hits = 0
    for i, row in enumerate(honest_rows, 1):
        query = row.title
        expected_cat = row.category

        emb = encode_query_text(query)
        raw_results = retriever.search(emb, pool_k)

        rank1_cat = ""
        query_price = 0.0
        query_meta: dict = {}
        if raw_results and rerank_cfg.enabled:
            top_aid = raw_results[0][0]
            top_aid = int(top_aid) if str(top_aid).isdigit() else top_aid
            top_meta = art_map.get(top_aid, {})
            rank1_cat = str(top_meta.get("category", ""))
            query_price = float(top_meta.get("price_inr") or 0.0)
            query_meta = top_meta

        use_rerank = rerank_cfg.enabled and bool(rank1_cat)
        if use_rerank:
            candidates = [
                (int(aid) if str(aid).isdigit() else aid, score) for aid, score in raw_results
            ]
            final_results = _rerank(
                candidates, query_price, rank1_cat, art_map, rerank_cfg, args.k,
                embeddings=None, query_meta=query_meta,
            )
        else:
            final_results = list(raw_results[: args.k])

        final_cats = [aid_to_cat.get(int(aid), "?") for aid, _ in final_results]
        hit = expected_cat in final_cats
        hits += hit
        aff_hit = hit or any(
            affinity_map.affinity(expected_cat, c) >= 0.4 for c in final_cats
        )
        affinity_hits += aff_hit

        if args.only_misses and hit:
            continue

        marker = "HIT " if hit else ("AFFINITY" if aff_hit else "MISS")
        print(f"[{i:02d}] {marker}  query={query!r}")
        print(f"       expected={expected_cat!r}  rank1_faiss_hit_category={rank1_cat!r}")
        print(f"       top{args.k}_categories_returned={final_cats}")
        print()

    print(f"Strict   recall@{args.k}: {hits}/{len(honest_rows)} = {hits / len(honest_rows):.4f}")
    print(
        f"Affinity recall@{args.k}: {affinity_hits}/{len(honest_rows)} = "
        f"{affinity_hits / len(honest_rows):.4f}  (>=0.4: exact + equivalent-group + related-group)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
