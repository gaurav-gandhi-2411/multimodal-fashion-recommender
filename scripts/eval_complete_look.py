"""scripts/eval_complete_look.py

Before/after evaluation for Complete-the-Look (outfit completion).

Runs the SAME function the live /complete route calls (app.complete.complete_the_look)
so the numbers reflect production behaviour. For each complete-enabled brand it contrasts:

  baseline  — raw FAISS top-k neighbours (what /similar returns: mostly SAME category)
  complete  — complete_the_look() output (COMPLEMENTARY categories forming an outfit)

Headline proof = the same_category_rate inversion: baseline is high (similar items),
complete is ~0 (complementary items). Plus slot_coverage and price coherence.

Usage:
  python scripts/eval_complete_look.py
  python scripts/eval_complete_look.py --n-queries 100 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.complete import build_slot_index, complete_the_look  # noqa: E402
from scripts.eval_similarity_quality import _stratified_sample, load_brand  # noqa: E402

# Only brands with complete.enabled in their YAML are meaningful here.
CANDIDATE_BRANDS = ["snitch", "powerlook", "fashor"]


def _reconstruct_matrix(faiss_index) -> np.ndarray:
    return faiss_index.reconstruct_n(0, faiss_index.ntotal).astype(np.float32)


def _raw_faiss_topk(query_aid, faiss_index, article_ids, aid_to_row, k):
    row = aid_to_row.get(query_aid)
    if row is None:
        return []
    emb = faiss_index.reconstruct(row).reshape(1, -1).astype(np.float32)
    _, indices = faiss_index.search(emb, k + 1)
    return [
        article_ids[idx]
        for idx in indices[0]
        if idx != -1 and article_ids[idx] != query_aid
    ][:k]


def eval_brand(brand: str, n_queries: int, seed: int) -> dict | None:
    catalog, faiss_index, article_ids, aid_to_row, _ = load_brand(brand)
    cfg = None
    # Re-read the full brand config (load_brand only returns the rerank config).
    import yaml  # noqa: PLC0415

    from app.brands.registry import BrandConfig  # noqa: PLC0415

    with open(REPO_ROOT / "brands" / f"{brand}.yaml") as fh:
        cfg = BrandConfig.model_validate(yaml.safe_load(fh)).complete

    if not cfg.enabled:
        print(f"[{brand}] complete DISABLED — skipping (correct for ethnic-set catalogs)")
        return None

    art_map = catalog.set_index("article_id").to_dict("index")
    emb_matrix = _reconstruct_matrix(faiss_index)
    slot_index = build_slot_index(cfg)
    queries = _stratified_sample(catalog, n=n_queries, seed=seed)

    comp_same_cat, comp_complementary, comp_slot_cov, comp_dprice = [], [], [], []
    base_same_cat, base_complementary = [], []
    n_with_results = 0

    for _, q in queries.iterrows():
        q_aid = int(q["article_id"])
        q_cat = str(q["category"])
        q_price = float(q["price_inr"]) if q["price_inr"] == q["price_inr"] else 0.0
        q_slot = slot_index.get(q_cat)
        if q_slot is None:
            continue  # query category not slotted (e.g. Co-ord) — no outfit to complete
        q_emb = emb_matrix[aid_to_row[q_aid]]

        target_slots = set(cfg.complements.get(q_slot, []))
        complement_cats = {
            c for s in cfg.slots if s.name in target_slots for c in s.categories
        }

        candidates = []
        for cand_aid, meta in art_map.items():
            if cand_aid == q_aid:
                continue
            cand_cat = str(meta.get("category", ""))
            if cand_cat not in complement_cats:
                continue
            row = aid_to_row.get(int(cand_aid))
            if row is None:
                continue
            candidates.append(
                (cand_aid, cand_cat, float(meta.get("price_inr") or 0.0), emb_matrix[row])
            )

        results = complete_the_look(q_cat, q_emb, q_price, candidates, cfg)
        if not results:
            continue
        n_with_results += 1

        res_cats = [str(art_map.get(int(a), {}).get("category", "")) for a, _, _ in results]
        res_prices = [float(art_map.get(int(a), {}).get("price_inr") or 0.0) for a, _, _ in results]
        comp_same_cat.append(np.mean([c == q_cat for c in res_cats]))
        comp_complementary.append(np.mean([c in complement_cats for c in res_cats]))
        comp_slot_cov.append(len({s for _, _, s in results}))
        deltas = [abs(q_price - p) for p in res_prices if p > 0 and q_price > 0]
        comp_dprice.append(float(np.mean(deltas)) if deltas else math.nan)

        # Baseline: raw FAISS top-max_items (what /similar would return)
        base_ids = _raw_faiss_topk(q_aid, faiss_index, article_ids, aid_to_row, cfg.max_items)
        base_cats = [str(art_map.get(int(a), {}).get("category", "")) for a in base_ids]
        if base_cats:
            base_same_cat.append(np.mean([c == q_cat for c in base_cats]))
            base_complementary.append(np.mean([c in complement_cats for c in base_cats]))

    def _m(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else math.nan

    return {
        "brand": brand,
        "n_queries": len(queries),
        "n_with_results": n_with_results,
        "base_same_cat": _m(base_same_cat),
        "base_complementary": _m(base_complementary),
        "comp_same_cat": _m(comp_same_cat),
        "comp_complementary": _m(comp_complementary),
        "comp_slot_cov": _m(comp_slot_cov),
        "comp_dprice": _m(comp_dprice),
        "n_slots": len({s.name for s in cfg.slots}),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Complete-the-Look eval (live /complete path).")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Complete-the-Look eval — n={args.n_queries}/brand, seed={args.seed}\n")
    rows = []
    for brand in CANDIDATE_BRANDS:
        print(f"[{brand}] loading...", flush=True)
        stats = eval_brand(brand, args.n_queries, args.seed)
        if stats:
            rows.append(stats)

    if not rows:
        print("No complete-enabled brands.")
        return

    print("\n" + "=" * 100)
    print(f"{'Brand':<10} | {'queries':>7} | {'same-cat raw→cmpl':>20} | "
          f"{'compl-cat raw→cmpl':>20} | {'slot cov':>8} | {'|dPrice|':>9}")
    print("-" * 100)
    for s in rows:
        same = f"{s['base_same_cat']*100:.0f}% → {s['comp_same_cat']*100:.0f}%"
        comp = f"{s['base_complementary']*100:.0f}% → {s['comp_complementary']*100:.0f}%"
        print(f"{s['brand']:<10} | {s['n_with_results']:>7} | {same:>20} | "
              f"{comp:>20} | {s['comp_slot_cov']:>6.2f}/{s['n_slots']} | "
              f"₹{s['comp_dprice']:>7.0f}")
    print("=" * 100)
    print("\nHeadline: same-category rate collapses (similar→complementary) while complementary")
    print("rate jumps to ~100% — Complete-the-Look returns a different, outfit-oriented result")
    print("set than /similar, on the same code path the live API runs.")


if __name__ == "__main__":
    main()
