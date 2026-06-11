"""
Find items per brand where rerank ON vs OFF produces a different top-5.
Outputs: top 5 items per brand with the clearest before/after difference,
         with ON vs OFF result tables side by side.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np

os.environ.update(SNITCH_API_KEY="demo", FASHOR_API_KEY="demo", POWERLOOK_API_KEY="demo", HM_API_KEY="demo")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.brands.registry import load_registry
from app.rerank import rerank as _do_rerank


def _emb(state, aid_int: int) -> np.ndarray | None:
    row = state.faiss_aid_to_row.get(aid_int)
    if row is None:
        return None
    return state.retriever.index.reconstruct(row).reshape(1, -1).astype(np.float32)


def top5(state, aid_int: int, rerank_on: bool) -> list[dict]:
    emb = _emb(state, aid_int)
    if emb is None:
        return []
    cfg = state.config.rerank
    pool_k = cfg.candidate_pool_size if rerank_on else 6
    raw = state.retriever.search(emb, k=pool_k)
    candidates = [(a, s) for a, s in raw if str(a) != str(aid_int)]
    if rerank_on:
        meta_q = state.art_map.get(aid_int, {})
        q_price = float(meta_q.get("price_inr") or 0.0)
        q_cat = str(meta_q.get("category", ""))
        candidates = _do_rerank(candidates, q_price, q_cat, state.art_map, cfg, k=5)
    else:
        candidates = candidates[:5]
    out = []
    for a, sc in candidates:
        key = int(a) if str(a).isdigit() else a
        m = state.art_map.get(key, {})
        out.append(dict(
            item_id=str(a),
            score=round(float(sc), 4),
            title=str(m.get("title", f"item {a}"))[:55],
            category=str(m.get("category", "?")),
            price=int(m["price_inr"]) if m.get("price_inr") is not None else None,
        ))
    return out


def rank_set(results: list[dict]) -> tuple[frozenset[str], list[str]]:
    ids = frozenset(r["item_id"] for r in results)
    order = [r["item_id"] for r in results]
    return ids, order


def score_diff(on_res: list[dict], off_res: list[dict]) -> float:
    on_ids, on_ord = rank_set(on_res)
    off_ids, off_ord = rank_set(off_res)
    # Items that changed set membership
    changed_set = len(on_ids.symmetric_difference(off_ids))
    # Rank displacement for common items
    common = on_ids & off_ids
    rank_delta = 0
    for item_id in common:
        r_on = on_ord.index(item_id)
        r_off = off_ord.index(item_id)
        rank_delta += abs(r_on - r_off)
    return changed_set * 3 + rank_delta  # weight set changes more


print("Loading registry...")
reg = load_registry("brands")
print()

for brand in ["snitch", "fashor", "powerlook"]:
    state = reg.get(brand)
    all_aids = list(state.art_map.keys())
    diffs: list[tuple[float, int, list[dict], list[dict]]] = []

    for aid in all_aids:
        if state.faiss_aid_to_row.get(aid) is None:
            continue
        on_res = top5(state, aid, rerank_on=True)
        off_res = top5(state, aid, rerank_on=False)
        if not on_res or not off_res:
            continue
        diff = score_diff(on_res, off_res)
        if diff > 0:
            diffs.append((diff, aid, on_res, off_res))

    diffs.sort(key=lambda x: x[0], reverse=True)
    top_items = diffs[:5]

    print(f"{'='*80}")
    print(f"BRAND: {brand.upper()}  -- {len(diffs)} items with non-trivial ON!=OFF (of {len(all_aids)} total)")
    print(f"{'='*80}")

    for diff_score, aid, on_res, off_res in top_items:
        meta_q = state.art_map.get(aid, {})
        q_cat = str(meta_q.get("category", "?"))
        q_price = meta_q.get("price_inr", "?")
        q_title = str(meta_q.get("title", f"item {aid}"))[:55]
        print(f"\n  Query aid={aid} [{q_cat}] Rs.{q_price}  diff_score={diff_score}")
        print(f"  Title: {q_title}")
        print(f"  {'OFF (raw FAISS)':<42}  {'ON (rerank)'}")
        print(f"  {'-'*42}  {'-'*42}")
        for i in range(5):
            off_r = off_res[i] if i < len(off_res) else {}
            on_r = on_res[i] if i < len(on_res) else {}
            off_str = f"{i+1}. [{off_r.get('category','?')[:12]:12}] Rs.{off_r.get('price','?'):5} {off_r.get('title','')[:25]}" if off_r else ""
            on_str  = f"{i+1}. [{on_r.get('category','?')[:12]:12}] Rs.{on_r.get('price','?'):5} {on_r.get('title','')[:25]}" if on_r else ""
            arrow = " <==" if on_r and off_r and on_r["item_id"] != off_r["item_id"] else "    "
            print(f"  {off_str:<42}{arrow}  {on_str}")
    print()
