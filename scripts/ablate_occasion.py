"""Isolate the marginal effect of the occasion boost (Feature #4).

For each brand, holds diversity/price/category fixed and sweeps w_occasion, reporting
strict cat-match (the INDEPENDENT relevance metric — not optimised by occasion) and
occasion-match rate among occasion-tagged queries. Same rerank() the live API calls.

Run: python scripts/ablate_occasion.py --n-queries 100 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.occasion import tag_occasions  # noqa: E402
from app.rerank import rerank as _rerank  # noqa: E402
from scripts.eval_similarity_quality import _stratified_sample, load_brand  # noqa: E402

BRANDS = ["snitch", "fashor", "powerlook"]


def _occ(meta, cfg) -> frozenset:
    lex = cfg.occasion_lexicon or None
    return tag_occasions(meta.get("title", ""), meta.get("description", ""), lex, cfg.parse_explicit_occasion)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for brand in BRANDS:
        catalog, faiss_index, article_ids, aid_to_row, cfg = load_brand(brand)
        art_map = catalog.set_index("article_id").to_dict("index")
        emb = faiss_index.reconstruct_n(0, faiss_index.ntotal).astype(np.float32)
        queries = _stratified_sample(catalog, n=args.n_queries, seed=args.seed)
        pool_k = cfg.candidate_pool_size
        current_w = cfg.w_occasion

        print(f"\n=== {brand}  (n={len(queries)}, diversity={cfg.w_diversity}, "
              f"current w_occasion={current_w}) ===")
        print(f"  {'w_occasion':>10} | {'strict':>7} | {'occ_match(tagged)':>18} | {'|dPrice|':>9}")

        for w_occ in (0.0, 0.08, 0.10, 0.15):
            strict_all, occ_all, dprice_all = [], [], []
            for _, q in queries.iterrows():
                q_aid = int(q["article_id"]); q_cat = str(q["category"])
                q_price = float(q["price_inr"]) if q["price_inr"] == q["price_inr"] else 0.0
                row = aid_to_row.get(q_aid)
                if row is None:
                    continue
                qvec = faiss_index.reconstruct(row).reshape(1, -1).astype(np.float32)
                scores, idxs = faiss_index.search(qvec, pool_k + 1)
                cands = [(article_ids[i], float(scores[0][j])) for j, i in enumerate(idxs[0])
                         if i != -1 and article_ids[i] != q_aid]
                q_meta = art_map.get(q_aid, {})
                embs = {a: emb[aid_to_row[a]] for a, _ in cands if a in aid_to_row}
                cfg.w_occasion = w_occ
                ranked = _rerank(cands, q_price, q_cat, art_map, cfg, args.k,
                                 embeddings=embs, query_meta=q_meta)
                metas = [art_map.get(int(a), {}) for a, _ in ranked]
                cats = [str(m.get("category", "")) for m in metas]
                strict_all.append(np.mean([c == q_cat for c in cats]) if cats else 0.0)
                prices = [float(m.get("price_inr") or 0.0) for m in metas]
                d = [abs(q_price - p) for p in prices if p > 0 and q_price > 0]
                if d:
                    dprice_all.append(float(np.mean(d)))
                q_occ = _occ(q_meta, cfg)
                if q_occ:  # only tagged queries count toward occasion_match
                    n_occ = [bool(q_occ & _occ(m, cfg)) for m in metas]
                    occ_all.append(np.mean(n_occ) if n_occ else 0.0)
            cfg.w_occasion = current_w  # restore
            strict = 100 * np.mean(strict_all)
            occ = 100 * np.mean(occ_all) if occ_all else float("nan")
            dprice = np.mean(dprice_all) if dprice_all else float("nan")
            print(f"  {w_occ:>10.2f} | {strict:6.0f}% | "
                  f"{occ:6.0f}% (n={len(occ_all):>3}) | ₹{dprice:>7.0f}")


if __name__ == "__main__":
    main()
