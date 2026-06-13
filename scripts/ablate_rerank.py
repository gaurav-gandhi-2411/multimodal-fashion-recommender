"""Ablation harness: isolate the marginal effect of each re-rank feature.

Honest measurement on the SAME code path the live /similar route uses
(app.rerank.rerank). For each brand and each config variant, reports:
  strict cat-match, mean |�+ABS+Price|, inter-dupe pairs at several cosine
  thresholds, distinct categories, same-band rate.

Also reports, for RAW FAISS top-k, the share of result sets that contain at
least one near-twin pair at thresholds 0.90/0.93/0.95/0.97 — i.e. the ceiling
on how much work a diversity re-ranker could possibly do on this catalog.

Run: python scripts/ablate_rerank.py --n-queries 100 --k 5
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rerank import CategoryAffinityMap, _price_band_index, rerank  # noqa: E402
from scripts.eval_similarity_quality import (  # noqa: E402
    _stratified_sample,
    load_brand,
)

BRANDS = ["snitch", "fashor", "powerlook"]
DUPE_THRESHOLDS = [0.90, 0.93, 0.95, 0.97]


def _reconstruct(faiss_index, row: int) -> np.ndarray:
    v = faiss_index.reconstruct(row).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _candidate_pool(query_aid, faiss_index, article_ids, aid_to_row, pool_k):
    row = aid_to_row.get(query_aid)
    if row is None:
        return [], {}
    emb = faiss_index.reconstruct(row).reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(emb, pool_k + 1)
    cands = [
        (article_ids[idx], float(scores[0][j]))
        for j, idx in enumerate(indices[0])
        if idx != -1 and article_ids[idx] != query_aid
    ]
    embs = {aid: _reconstruct(faiss_index, aid_to_row[aid]) for aid, _ in cands}
    return cands, embs


def _max_pairwise_cos(aids: list, embs: dict) -> float:
    vecs = [embs[a] for a in aids if a in embs]
    best = 0.0
    for a, b in itertools.combinations(vecs, 2):
        best = max(best, float(a @ b))
    return best


def _inter_dupe_pairs(aids: list, embs: dict, thr: float) -> int:
    vecs = [embs[a] for a in aids if a in embs]
    return sum(1 for a, b in itertools.combinations(vecs, 2) if float(a @ b) >= thr)


def _variant_config(base_cfg, *, w_diversity, dupe, w_band):
    cfg = base_cfg.model_copy(deep=True)
    cfg.w_diversity = w_diversity
    cfg.dupe_sim_threshold = dupe
    cfg.w_price_band = w_band
    return cfg


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for brand in BRANDS:
        catalog, faiss_index, article_ids, aid_to_row, base_cfg = load_brand(brand)
        art_map = catalog.set_index("article_id").to_dict("index")
        aff = CategoryAffinityMap(base_cfg)
        queries = _stratified_sample(catalog, n=args.n_queries, seed=args.seed)
        pool_k = base_cfg.candidate_pool_size
        bands = base_cfg.price_bands_inr

        variants = {
            "base (div0 band0)": _variant_config(base_cfg, w_diversity=0.0, dupe=0.97, w_band=0.0),
            "band only":         _variant_config(base_cfg, w_diversity=0.0, dupe=0.97, w_band=0.05),
            "div .15@0.97":      _variant_config(base_cfg, w_diversity=0.15, dupe=0.97, w_band=0.0),
            "div .15@0.92":      _variant_config(base_cfg, w_diversity=0.15, dupe=0.92, w_band=0.0),
            "div .25@0.90":      _variant_config(base_cfg, w_diversity=0.25, dupe=0.90, w_band=0.0),
        }

        # accumulators
        agg = {name: {"strict": [], "dprice": [], "dupe": [], "distinct": [], "band": []}
               for name in variants}
        raw_twin_share = {t: 0 for t in DUPE_THRESHOLDS}
        n_used = 0

        for _, q in queries.iterrows():
            q_aid = int(q["article_id"])
            q_cat = str(q["category"])
            q_price = float(q["price_inr"]) if q["price_inr"] == q["price_inr"] else 0.0
            cands, embs = _candidate_pool(q_aid, faiss_index, article_ids, aid_to_row, pool_k)
            if not cands:
                continue
            n_used += 1

            # raw top-k near-twin ceiling
            raw_ids = [a for a, _ in cands[: args.k]]
            mx = _max_pairwise_cos(raw_ids, embs)
            for t in DUPE_THRESHOLDS:
                if mx >= t:
                    raw_twin_share[t] += 1

            q_band = _price_band_index(q_price, bands) if bands else -1
            for name, cfg in variants.items():
                ranked = rerank(cands, q_price, q_cat, art_map, cfg, args.k, embeddings=embs)
                ids = [a for a, _ in ranked]
                metas = [art_map.get(int(a) if str(a).isdigit() else a, {}) for a in ids]
                cats = [str(m.get("category", "")) for m in metas]
                prices = [float(m.get("price_inr") or 0.0) for m in metas]
                strict = np.mean([c == q_cat for c in cats]) if cats else 0.0
                dprice = np.mean([abs(q_price - p) for p in prices if p > 0 and q_price > 0]) \
                    if any(p > 0 for p in prices) else np.nan
                dupe = _inter_dupe_pairs(ids, embs, 0.92)
                distinct = len(set(cats))
                if bands:
                    band_hits = [(_price_band_index(p, bands) == q_band) for p in prices if p > 0]
                    band_rate = np.mean(band_hits) if band_hits else np.nan
                else:
                    band_rate = np.nan
                agg[name]["strict"].append(strict)
                agg[name]["dprice"].append(dprice)
                agg[name]["dupe"].append(dupe)
                agg[name]["distinct"].append(distinct)
                agg[name]["band"].append(band_rate)

        print(f"\n=== {brand}  (n={n_used}, k={args.k}, pool={pool_k}, bands={bands}) ===")
        print("  RAW top-k near-twin share (>=1 pair at threshold):")
        for t in DUPE_THRESHOLDS:
            print(f"    cos>={t}: {raw_twin_share[t]}/{n_used} "
                  f"({100*raw_twin_share[t]/max(n_used,1):.0f}%)")
        print(f"  {'variant':<18} | {'strict':>7} | {'|dPrice|':>9} | "
              f"{'dupe@.92':>9} | {'distinct':>8} | {'band%':>6}")
        for name in variants:
            a = agg[name]
            strict = 100 * np.nanmean(a["strict"])
            dprice = np.nanmean(a["dprice"])
            dupe = np.nanmean(a["dupe"])
            distinct = np.nanmean(a["distinct"])
            band = 100 * np.nanmean(a["band"]) if not np.all(np.isnan(a["band"])) else float("nan")
            print(f"  {name:<18} | {strict:6.0f}% | {dprice:8.0f} | "
                  f"{dupe:9.2f} | {distinct:8.2f} | {band:5.0f}%")


if __name__ == "__main__":
    main()
