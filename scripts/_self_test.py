"""Self-test: run /similar logic in-process for all brands and produce a raw data dump."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.update(
    SNITCH_API_KEY="demo",
    FASHOR_API_KEY="demo",
    POWERLOOK_API_KEY="demo",
    HM_API_KEY="demo",
)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.brands.registry import load_registry
from app.rerank import rerank as _do_rerank

print("Loading registry …", flush=True)
reg = load_registry("brands")
print(f"Loaded: {sorted(reg.brand_names())}\n", flush=True)


# ---------------------------------------------------------------------------
# Mirror of routes.py _get_item_embedding + /similar logic
# ---------------------------------------------------------------------------

def _emb(state, item_id_str: str) -> np.ndarray | None:
    try:
        aid = int(item_id_str)
    except ValueError:
        return None
    row = state.faiss_aid_to_row.get(aid)
    if row is None:
        return None
    return state.retriever.index.reconstruct(row).reshape(1, -1).astype(np.float32)


def similar(state, item_id_str: str, k: int = 5, rerank_on: bool = True) -> list[dict]:
    emb = _emb(state, item_id_str)
    if emb is None:
        return []
    cfg = state.config.rerank
    pool_k = cfg.candidate_pool_size if rerank_on else k + 1
    raw = state.retriever.search(emb, k=pool_k)
    candidates = [(aid, sc) for aid, sc in raw if str(aid) != item_id_str]
    if rerank_on:
        q_aid = int(item_id_str)
        meta = state.art_map.get(q_aid, {})
        q_price = float(meta.get("price_inr") or 0.0)
        q_cat = str(meta.get("category", ""))
        candidates = _do_rerank(candidates, q_price, q_cat, state.art_map, cfg, k)
    else:
        candidates = candidates[:k]
    out = []
    for aid, sc in candidates:
        # article_ids from retriever may be str or int — normalise
        m = state.art_map.get(int(aid) if str(aid).isdigit() else aid, {})
        out.append(
            dict(
                item_id=str(aid),
                score=round(float(sc), 4),
                title=str(m.get("title", f"item {aid}"))[:60],
                category=str(m.get("category", "?")),
                price=int(m["price_inr"]) if m.get("price_inr") is not None else None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Query items (chosen to span categories; A/B items chosen where both
# price and category-affinity levers are most visible)
# ---------------------------------------------------------------------------

QUERIES: dict[str, list[str]] = {
    # Shirts, T-Shirts, Trousers, Jeans, Jackets, Shorts — 6 distinct cats
    "snitch": ["1", "51", "101", "151", "251", "451"],
    # 3P Kurta Set, 2P Kurta Set, Kurtas, Dresses, Co-ord Set, Kurti/Tunics
    "fashor": ["86", "84", "731", "789", "4", "868"],
    # Shirt, T-Shirt, Bottom, Vest, Jacket, Track-Suit
    "powerlook": ["3", "1", "17", "27", "589", "113"],
}

# One A/B item per brand:
#   snitch 101  — Trouser ₹1019; Snitch price window ₹500–₹3k → high penalty spread
#   fashor 4    — Co-ord Set ₹2249; ethnic_set equiv-group + high price weight (0.25)
#   powerlook 17 — Bottom ₹1199; only price lever (no category groups)
AB_ITEMS: dict[str, str] = {
    "snitch": "101",
    "fashor": "4",
    "powerlook": "17",
}


# ---------------------------------------------------------------------------
# Run all queries
# ---------------------------------------------------------------------------

results: dict = {}

for brand, aids in QUERIES.items():
    print(f"Running {brand} …", flush=True)
    state = reg.get(brand)
    brand_results = {}
    for aid in aids:
        on = similar(state, aid, k=5, rerank_on=True)
        brand_results[aid] = {"on": on}
    # A/B
    ab_aid = AB_ITEMS[brand]
    off = similar(state, ab_aid, k=5, rerank_on=False)
    brand_results[ab_aid]["off"] = off
    results[brand] = brand_results

print("\nDone — dumping JSON\n")
print(json.dumps(results, indent=2, ensure_ascii=False))
