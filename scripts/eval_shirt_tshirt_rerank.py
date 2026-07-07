"""scripts/eval_shirt_tshirt_rerank.py

Prototype eval for the Shirt/T-Shirt cross-brand visual-search ambiguity found during
the 2026-07-07 audit: a genuinely out-of-catalog t-shirt image query on Powerlook
returns Shirt-category results, because (a) CLIP-512 embedding space genuinely favors
Shirt matches for plain, unpatterned tees against this catalog's composition, and
(b) the existing top-1-category-anchor rerank (app/api/routes.py) reinforces whichever
category the anchor happens to be, with ZERO affinity credit for the other category
since Powerlook's `category_groups` is empty.

Tests whether declaring {Shirt, T-Shirt} as a "related" category group (using the
already-existing related_group_bonus machinery, no new code) changes category-match@5
for a cross-brand query set, without regressing self-match (in-catalog) queries.

Runs entirely locally against indices/powerlook/visual.faiss + data/powerlook/items.parquet
-- no live API calls, no backend redeploy needed to prototype.

RESULT (2026-07-07): NEGATIVE. Do not ship.
  - related_group_bonus=0.40: 0/5 change on the diagnosed failing query (still 5/5 Shirt).
  - Even equivalent_group_bonus=0.70 (max available, near-merging the categories): still
    5/5 Shirt on the failing query.
  - The same equivalent-level config, applied broadly, regressed a clean in-catalog
    self-match strict@5 from 100% (50/50) to 96% (48/50) on a 10-item Shirt/T-Shirt sample.
  - Root cause: Powerlook has 571 Shirt items vs 229 T-Shirt items. In the raw CLIP-512
    ranking for the failing query, T-Shirt candidates ARE competitively interspersed near
    the top (rank 2 of 50, raw sim 0.7563 vs rank-1 Shirt's 0.7585) -- but there are simply
    many more high-scoring exact-match Shirt candidates than T-Shirt candidates, so category
    affinity (scored per-candidate, not per-category-quota) can't out-vote the raw count
    advantage without merging the categories outright (which breaks clean cases elsewhere).
  - Conclusion: this needs either (a) a per-category top-N quota / interleaving mechanism
    (structurally different from score-based affinity -- not attempted here, candidate for
    a future pass) or (b) a better embedding that separates knit-tee vs woven-shirt texture
    more cleanly at the raw-similarity level. See PROJECT_MEMORY / audit report for the
    embedding-model A/B proposal this evidence feeds into.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import io
import requests

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.faiss_index import FaissRetriever  # noqa: E402
from app.rerank import RerankConfig, CategoryGroupConfig, rerank  # noqa: E402

K = 5
POOL_K = 50


def _load_clip():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    return model, preprocess


def _encode(model, preprocess, img_bytes: bytes) -> np.ndarray:
    import torch
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).numpy().astype(np.float32)


def build_art_map(cat: pd.DataFrame) -> dict:
    art_map = {}
    for _, row in cat.iterrows():
        art_map[int(row["article_id"])] = {
            "category": row["category"],
            "price_inr": row["price_inr"],
            "title": row.get("title", ""),
            "description": row.get("description", ""),
        }
    return art_map


def run_query(retriever, art_map, model, preprocess, img_bytes, base_cfg, fixed_cfg):
    emb = _encode(model, preprocess, img_bytes)
    raw = retriever.search(emb, POOL_K)
    raw_int = [(int(aid), score) for aid, score in raw]

    top_aid = raw_int[0][0]
    top_cat = art_map.get(top_aid, {}).get("category", "")
    top_price = art_map.get(top_aid, {}).get("price_inr", 0.0)

    baseline = rerank(raw_int, top_price, top_cat, art_map, base_cfg, K, embeddings=None)
    fixed = rerank(raw_int, top_price, top_cat, art_map, fixed_cfg, K, embeddings=None)
    return top_cat, baseline, fixed


def main() -> None:
    cat = pd.read_parquet(REPO_ROOT / "data/powerlook/items.parquet")
    art_map = build_art_map(cat)

    retriever = FaissRetriever.load(str(REPO_ROOT / "indices/powerlook/visual.faiss"))
    print(f"Loaded Powerlook visual index: {retriever.index.ntotal} items")

    print("Loading CLIP...")
    model, preprocess = _load_clip()

    # Baseline: current live Powerlook rerank config (category_groups empty).
    base_cfg = RerankConfig(
        enabled=True, candidate_pool_size=POOL_K, w_similarity=0.70, w_price_penalty=0.10,
        w_category_affinity=0.15, price_norm_inr=400.0, equivalent_group_bonus=0.70,
        related_group_bonus=0.40, category_groups=[], w_diversity=0.0,  # MMR off for a clean A/B
        w_occasion=0.0,
    )
    # Fix: declare Shirt/T-Shirt as a related pair (no other config changed).
    fixed_cfg = base_cfg.model_copy(deep=True)
    fixed_cfg.category_groups = [
        CategoryGroupConfig(name="shirts", members=["Shirt"], related_groups=["tshirts"]),
        CategoryGroupConfig(name="tshirts", members=["T-Shirt"], related_groups=["shirts"]),
    ]

    # Cross-brand query set: sample real catalog images from Snitch + Fashor
    # (out-of-catalog for Powerlook -- this is where the bug manifests; Powerlook's
    # own self-queries already work perfectly and are not the failure mode).
    snitch = pd.read_parquet(REPO_ROOT / "data/snitch/items.parquet")
    fashor_df = pd.read_parquet(REPO_ROOT / "data/fashor/items.parquet")
    snitch_tees = snitch[snitch["category"] == "T-Shirts"].sample(10, random_state=42)
    snitch_shirts = snitch[snitch["category"] == "Shirts"].sample(10, random_state=42)

    results = []
    for label, sample in [("snitch_tshirt", snitch_tees), ("snitch_shirt", snitch_shirts)]:
        for _, row in sample.iterrows():
            try:
                img_bytes = requests.get(row["image_url"], timeout=15).content
            except Exception as e:
                print(f"  skip {row['article_id']}: {e}")
                continue
            true_type = "T-Shirt" if label == "snitch_tshirt" else "Shirt"
            top_cat, base_res, fixed_res = run_query(
                retriever, art_map, model, preprocess, img_bytes, base_cfg, fixed_cfg
            )
            base_cats = [art_map.get(aid, {}).get("category", "?") for aid, _ in base_res]
            fixed_cats = [art_map.get(aid, {}).get("category", "?") for aid, _ in fixed_res]
            base_match = sum(1 for c in base_cats if c == true_type)
            fixed_match = sum(1 for c in fixed_cats if c == true_type)
            results.append({
                "query_aid": row["article_id"], "true_type": true_type, "anchor_cat": top_cat,
                "base_match": base_match, "fixed_match": fixed_match,
                "base_cats": base_cats, "fixed_cats": fixed_cats,
            })
            print(f"aid={row['article_id']:<6} true={true_type:8s} anchor={top_cat:8s} "
                  f"base@5={base_match}/5 fixed@5={fixed_match}/5  "
                  f"base={base_cats} fixed={fixed_cats}")

    n = len(results)
    base_total = sum(r["base_match"] for r in results)
    fixed_total = sum(r["fixed_match"] for r in results)
    print(f"\n=== Powerlook cross-brand category-match@5 (n={n} queries, {n*5} slots) ===")
    print(f"Baseline (no Shirt~T-Shirt relation): {base_total}/{n*5} = {base_total/(n*5):.1%}")
    print(f"Fixed (Shirt~T-Shirt related, 0.40):  {fixed_total}/{n*5} = {fixed_total/(n*5):.1%}")


if __name__ == "__main__":
    main()
