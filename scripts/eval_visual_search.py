"""scripts/eval_visual_search.py -- Visual-search quality eval (pure CLIP-512 path).

Per brand (snitch, fashor, powerlook):
  1. Load the brand's per-brand visual FAISS index from indices/{brand}/visual.faiss
     via FaissRetriever.  Skip brand with a clear message if the directory is absent.
  2. Load data/{brand}/items.parquet and the brand's YAML to get art_map.
  3. Sample N=50 catalog items WITH images (seed=42, stratified across categories).
  4. Encode each item's image via encode_query_image() -- the same pure CLIP-512 path
     the /visual-search route uses (no tower, no SBERT).
  5. Search the visual FAISS index; report:
       self_retrieval_rate  -- item's own id appears in top-k (should be ~100% now)
       category_match_rate@5 -- fraction of top-5 whose category == query category
  6. Print a per-brand summary table.

Usage:
    python scripts/eval_visual_search.py
    python scripts/eval_visual_search.py --n-queries 50 --k 10 --seed 42
    python scripts/eval_visual_search.py --brands snitch
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.visual import encode_query_image  # noqa: E402
from src.retrieval.faiss_index import FaissRetriever  # noqa: E402

BRANDS: list[str] = ["snitch", "fashor", "powerlook"]


# ---------------------------------------------------------------------------
# Image filename resolution
# ---------------------------------------------------------------------------


def _image_path(brand: str, row: pd.Series) -> Path | None:
    """Return the local image path for a catalog row, or None if absent.

    Convention (all three brands): data/{brand}/images/{product_id}.jpg.
    Falls back to {article_id}.jpg in case product_id is missing.
    """
    images_dir = REPO_ROOT / "data" / brand / "images"
    pid = str(row.get("product_id", "")).strip()
    if pid:
        candidate = images_dir / f"{pid}.jpg"
        if candidate.exists():
            return candidate
    aid = str(row.get("article_id", "")).strip()
    if aid:
        candidate = images_dir / f"{aid}.jpg"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Sampling (stratified, seed=42)
# ---------------------------------------------------------------------------


def _stratified_sample(catalog: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows spread across categories (proportional; min-1 per category)."""
    rng = np.random.default_rng(seed)
    cat_counts = catalog["category"].value_counts()
    n_cats = len(cat_counts)

    if n_cats >= n:
        selected = []
        for cat in cat_counts.head(n).index:
            rows = catalog[catalog["category"] == cat]
            selected.append(rows.sample(1, random_state=int(rng.integers(1_000_000))))
        return pd.concat(selected).reset_index(drop=True)

    slots: dict[str, int] = {cat: 1 for cat in cat_counts.index}
    remaining = n - n_cats
    if remaining > 0:
        proportional = (cat_counts / cat_counts.sum() * remaining).round().astype(int)
        for cat, extra in proportional.items():
            max_extra = len(catalog[catalog["category"] == cat]) - slots[cat]
            slots[cat] += min(int(extra), max_extra)

    total = sum(slots.values())
    for cat in reversed(cat_counts.index.tolist()):
        if total <= n:
            break
        if slots[cat] > 1:
            slots[cat] -= 1
            total -= 1

    samples = []
    for cat, count in slots.items():
        rows = catalog[catalog["category"] == cat]
        n_sample = min(count, len(rows))
        samples.append(rows.sample(n_sample, random_state=int(rng.integers(1_000_000))))

    return pd.concat(samples).head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------


@dataclass
class VisualQueryResult:
    brand: str
    article_id: int
    category: str
    top_k_aids: list[int]
    top_k_cats: list[str]
    category_match_rate_at5: float  # fraction of top-5 whose category == query category
    self_retrieved: bool             # source item appears in top-k


# ---------------------------------------------------------------------------
# Per-brand eval
# ---------------------------------------------------------------------------


def eval_brand(
    brand: str,
    n_queries: int = 50,
    k: int = 10,
    seed: int = 42,
) -> tuple[list[VisualQueryResult], int, int]:
    """Evaluate visual search quality for one brand using the visual FAISS index.

    Returns:
        results: per-query VisualQueryResult list (usable queries only)
        n_sampled: number of items drawn from the stratified sample
        n_usable: number of sampled items that had a local image file
    """
    visual_index_dir = REPO_ROOT / "indices" / brand / "visual.faiss"
    if not visual_index_dir.is_dir():
        print(
            f"  [SKIP] Visual index not found at {visual_index_dir}. "
            "Run scripts/build_visual_index.py first."
        )
        return [], 0, 0

    print(f"  Loading visual index from {visual_index_dir}...", end=" ", flush=True)
    retriever = FaissRetriever.load(str(visual_index_dir))
    index_article_ids: list[int] = [int(aid) for aid in retriever.article_ids]
    index_aid_set: set[int] = set(index_article_ids)
    print(f"{retriever.index.ntotal} vectors, dim={retriever.index.d}")

    catalog_path = REPO_ROOT / "data" / brand / "items.parquet"
    catalog = pd.read_parquet(catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)
    art_map: dict[int, dict] = catalog.set_index("article_id").to_dict("index")
    print(f"  Catalog: {len(catalog)} items, {catalog['category'].nunique()} categories")

    # Restrict sample to items present in the visual index (have images).
    indexed_catalog = catalog[catalog["article_id"].isin(index_aid_set)]
    sample = _stratified_sample(indexed_catalog, n=n_queries, seed=seed)
    n_sampled = len(sample)

    results: list[VisualQueryResult] = []
    n_skipped = 0

    for _, row in sample.iterrows():
        img_path = _image_path(brand, row)
        if img_path is None:
            n_skipped += 1
            continue

        q_aid = int(row["article_id"])
        q_cat = str(row["category"])

        img_bytes = img_path.read_bytes()
        try:
            # Pure CLIP-512 path -- exactly the /visual-search route encoding.
            query_emb = encode_query_image(img_bytes)
        except ValueError:
            n_skipped += 1
            continue

        # FAISS search -- fetch k+1 so self-retrieval check is unambiguous at k.
        raw = retriever.search(query_emb, k + 1)
        top_aids = [int(aid) for aid, _ in raw]
        top_aids_k = top_aids[:k]

        top_k_cats = [str(art_map.get(aid, {}).get("category", "")) for aid in top_aids_k]

        # category_match_rate@5 (cap at top-5 for comparability)
        top5 = top_k_cats[:5]
        cat_match_at5 = sum(1 for c in top5 if c == q_cat) / len(top5) if top5 else 0.0

        self_ret = q_aid in top_aids_k

        results.append(VisualQueryResult(
            brand=brand,
            article_id=q_aid,
            category=q_cat,
            top_k_aids=top_aids_k,
            top_k_cats=top_k_cats,
            category_match_rate_at5=cat_match_at5,
            self_retrieved=self_ret,
        ))

    n_usable = n_sampled - n_skipped
    return results, n_sampled, n_usable


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_table(all_results: dict[str, tuple[list[VisualQueryResult], int, int]]) -> None:
    col_brand = 12
    col_usable = 22
    col_catmatch = 32
    col_selfret = 22

    header = (
        f"{'Brand':<{col_brand}} | "
        f"{'Usable images':<{col_usable}} | "
        f"{'Category match rate @5':<{col_catmatch}} | "
        f"{'Self-retrieval rate':<{col_selfret}}"
    )
    sep = "-" * len(header)
    print()
    print("Visual Search Eval -- pure CLIP-512 against visual.faiss index")
    print("  category_match_rate@5: fraction of top-5 results whose category == query category")
    print("  self_retrieval_rate:   fraction of queries where source item appears in top-k")
    print(sep)
    print(header)
    print(sep)

    for brand in BRANDS:
        if brand not in all_results:
            continue
        results, n_sampled, n_usable = all_results[brand]
        if not results:
            usable_str = f"0 / {n_sampled} (0 usable)"
            print(
                f"{brand:<{col_brand}} | "
                f"{usable_str:<{col_usable}} | "
                f"{'--':<{col_catmatch}} | "
                f"{'--':<{col_selfret}}"
            )
            continue

        cat_rates = [r.category_match_rate_at5 for r in results]
        self_rates = [1.0 if r.self_retrieved else 0.0 for r in results]
        mean_cat = float(np.mean(cat_rates))
        mean_self = float(np.mean(self_rates))

        usable_str = f"{n_usable} / {n_sampled} usable"
        cat_str = f"{mean_cat * 100:.1f}%  (mean over {len(results)} queries)"
        self_str = f"{mean_self * 100:.1f}%"

        print(
            f"{brand:<{col_brand}} | "
            f"{usable_str:<{col_usable}} | "
            f"{cat_str:<{col_catmatch}} | "
            f"{self_str:<{col_selfret}}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual-search quality eval -- pure CLIP-512 path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-queries", type=int, default=50, metavar="N",
                   help="Items sampled per brand (after image filter)")
    p.add_argument("--k", type=int, default=10, metavar="K",
                   help="FAISS top-k")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--brands",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=BRANDS,
        metavar="BRAND1,BRAND2,...",
        help="Comma-separated brand slugs to evaluate",
    )
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    print(
        f"Visual Search Eval (pure CLIP-512) -- brands: {args.brands}, "
        f"n_queries={args.n_queries}, k={args.k}, seed={args.seed}"
    )

    all_results: dict[str, tuple[list[VisualQueryResult], int, int]] = {}
    for brand in args.brands:
        print(f"\n[{brand}]")
        results, n_sampled, n_usable = eval_brand(
            brand, n_queries=args.n_queries, k=args.k, seed=args.seed
        )
        if n_sampled > 0:
            print(
                f"  Sampled: {n_sampled}  Usable (have local image): {n_usable}  "
                f"Skipped (no image): {n_sampled - n_usable}"
            )
        if results:
            mean_cat = float(np.mean([r.category_match_rate_at5 for r in results]))
            mean_self = float(np.mean([1.0 if r.self_retrieved else 0.0 for r in results]))
            print(f"  category_match_rate@5: {mean_cat * 100:.1f}%")
            print(f"  self_retrieval_rate:   {mean_self * 100:.1f}%")
        all_results[brand] = (results, n_sampled, n_usable)

    print_table(all_results)


if __name__ == "__main__":
    main()
