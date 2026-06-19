"""scripts/eval_color_rerank.py

Offline A/B evaluation of color-aware reranking for /visual-search.

Method:
  For each brand, sample N query items (stratified by color bucket).
  For each query item:
    1. Retrieve its CLIP embedding from the visual FAISS index.
    2. Run FAISS search to get top-k raw results (simulates CLIP-only ranking).
    3. Apply color_rerank() to produce color-boosted results.
    4. For each result, look up its HSV color from the color index.
    5. Compute color_similarity(query_color, result_color) for each result.
  Report: mean_color_sim@5 before vs after color reranking.

Interpretation:
  Higher mean_color_sim@5 after = color rerank is surfacing more color-coherent items.
  Trivially high scores in both conditions = color index is too homogeneous to measure.

Usage:
  python scripts/eval_color_rerank.py
  python scripts/eval_color_rerank.py --n-queries 50 --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.color import ColorIndex, color_rerank, color_similarity, load_color_index


def _load_brand_visual_index(brand_cfg: dict) -> tuple[faiss.Index, list[int]] | None:
    """Load brand visual FAISS index. Returns (index, article_ids) or None."""
    vis_path = brand_cfg.get("visual_index_path")
    if not vis_path:
        return None
    p = Path(vis_path)
    idx_file = p / "faiss.index"
    ids_file = p / "article_ids.pkl"
    if not idx_file.exists() or not ids_file.exists():
        return None
    import pickle
    index = faiss.read_index(str(idx_file))
    with ids_file.open("rb") as fh:
        article_ids: list[int] = pickle.load(fh)  # noqa: S301
    return index, article_ids


def _eval_brand(
    brand_slug: str,
    brand_cfg: dict,
    color_index: ColorIndex,
    *,
    n_queries: int,
    k: int,
    rng: np.random.Generator,
) -> dict:
    result = {
        "brand": brand_slug,
        "n_queries": 0,
        "mean_color_sim_before": None,
        "mean_color_sim_after": None,
        "delta": None,
    }

    loaded = _load_brand_visual_index(brand_cfg)
    if loaded is None:
        print(f"  {brand_slug}: no visual index found — skipping")
        return result

    index, article_ids = loaded
    n_total = index.ntotal

    # Only query items that are in the color index (so we have ground-truth color)
    indexed_items = [
        (i, int(article_ids[i]))
        for i in range(n_total)
        if str(article_ids[i]) in color_index
    ]
    if not indexed_items:
        print(f"  {brand_slug}: no items in both FAISS + color index — skipping")
        return result

    n_queries = min(n_queries, len(indexed_items))
    sample_idx = rng.choice(len(indexed_items), size=n_queries, replace=False)
    queries = [indexed_items[i] for i in sample_idx]

    sims_before: list[float] = []
    sims_after: list[float] = []

    for row_idx, query_aid in queries:
        query_hsv = color_index.get(str(query_aid))
        if query_hsv is None:
            continue

        # Retrieve query embedding
        query_emb = index.reconstruct(row_idx).astype(np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1.0)
        query_batch = query_emb.reshape(1, -1)

        # FAISS search (pool_k to allow self-exclusion + have enough candidates)
        pool_k = min(k * 3 + 1, n_total)
        scores_mat, ids_mat = index.search(query_batch, pool_k)
        raw: list[tuple[int, float]] = [
            (int(article_ids[ids_mat[0][i]]), float(scores_mat[0][i]))
            for i in range(pool_k)
            if ids_mat[0][i] != row_idx  # exclude self
        ][:k]

        if not raw:
            continue

        # Color rerank
        reranked = color_rerank(raw, color_index, query_hsv)

        # Compute mean color_sim@k before and after
        def mean_color_sim(result_list: list[tuple[int | str, float]]) -> float:
            sims = []
            for aid, _ in result_list[:k]:
                item_hsv = color_index.get(str(aid))
                if item_hsv:
                    sims.append(color_similarity(query_hsv, item_hsv))
            return float(np.mean(sims)) if sims else 0.0

        sims_before.append(mean_color_sim(raw))
        sims_after.append(mean_color_sim(reranked))

    if not sims_before:
        return result

    result.update({
        "n_queries": len(sims_before),
        "mean_color_sim_before": round(float(np.mean(sims_before)), 4),
        "mean_color_sim_after": round(float(np.mean(sims_after)), 4),
        "delta": round(float(np.mean(sims_after)) - float(np.mean(sims_before)), 4),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline A/B eval for color-aware reranking")
    parser.add_argument("--n-queries", type=int, default=50, dest="n_queries")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    brands_dir = REPO_ROOT / "brands"

    print(f"\nColor rerank eval  n_queries={args.n_queries}  k={args.k}  seed={args.seed}\n")
    print(f"{'Brand':<12} {'N':<6} {'Before':>10} {'After':>10} {'Delta':>10}  Verdict")
    print("-" * 60)

    for yaml_path in sorted(brands_dir.glob("*.yaml")):
        with yaml_path.open() as fh:
            cfg = yaml.safe_load(fh)
        brand = cfg.get("brand", yaml_path.stem)
        color_idx_path = cfg.get("color_index_path")
        if not color_idx_path:
            print(f"  {brand}: no color_index_path configured — skipping")
            continue
        color_index = load_color_index(REPO_ROOT / color_idx_path)
        res = _eval_brand(brand, cfg, color_index, n_queries=args.n_queries, k=args.k, rng=rng)

        if res["mean_color_sim_before"] is None:
            verdict = "SKIPPED"
        elif res["delta"] is None:
            verdict = "SKIPPED"
        elif res["delta"] >= 0.01:
            verdict = "IMPROVED ✓"
        elif res["delta"] >= 0.0:
            verdict = "NEUTRAL"
        else:
            verdict = "DEGRADED ✗"

        before_str = (
            f"{res['mean_color_sim_before']:.4f}"
            if res["mean_color_sim_before"] is not None
            else "N/A"
        )
        after_str = (
            f"{res['mean_color_sim_after']:.4f}"
            if res["mean_color_sim_after"] is not None
            else "N/A"
        )
        delta_str = f"{res['delta']:+.4f}" if res["delta"] is not None else "N/A"
        print(
            f"{brand:<12} {res['n_queries']:<6} {before_str:>10} {after_str:>10}"
            f" {delta_str:>10}  {verdict}"
        )

    print(
        "\nInterpretation: delta >= 0.01 = color rerank measurably improves color coherence at top-k."
    )
    print("Run scripts/build_visual_index.py first if indices are missing.\n")


if __name__ == "__main__":
    main()
