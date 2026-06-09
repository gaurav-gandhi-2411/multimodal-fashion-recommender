"""
scripts/eval_similarity_quality.py

One-off similarity-quality inspection for Indian brand catalogs. Runs the
same item-to-item retrieval as /similar without touching the API, model
checkpoints, encoders, or FAISS build logic.

For each brand (snitch, fashor, powerlook):
  1. Samples 25 query items spread across categories (stratified).
  2. Retrieves top-5 similar items (same logic as /similar route).
  3. Computes: category-match rate, mean |ΔPrice| (INR), self-match / near-dupe flags.
  4. Dumps a browsable HTML report.
  5. Flags worst-10 results (lowest category-match) for failure-mode inspection.

Usage:
  python scripts/eval_similarity_quality.py
  python scripts/eval_similarity_quality.py --output reports/similarity_eval.html
  python scripts/eval_similarity_quality.py --n-queries 25 --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

BRANDS: list[str] = ["snitch", "fashor", "powerlook"]
NEAR_DUPE_THRESHOLD = 0.995  # cosine-sim above this (non-self) flagged as near-dupe
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "similarity_eval.html"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NeighborResult:
    article_id: int
    title: str
    category: str
    price_inr: float
    image_url: str
    pdp_url: str
    score: float
    is_category_match: bool
    is_self: bool
    is_near_dupe: bool


@dataclass
class QueryResult:
    brand: str
    query_article_id: int
    query_title: str
    query_category: str
    query_price_inr: float
    query_image_url: str
    query_pdp_url: str
    neighbors: list[NeighborResult]
    category_match_rate: float
    mean_price_delta_inr: float
    has_self_match: bool
    n_near_dupes: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_faiss(index_dir: Path) -> tuple[faiss.Index, list[int]]:
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "article_ids.pkl", "rb") as fh:
        raw_ids = pickle.load(fh)
    return index, [int(aid) for aid in raw_ids]


def load_brand(brand: str) -> tuple[pd.DataFrame, faiss.Index, list[int], dict[int, int]]:
    """Load catalog + FAISS index. No API key or model checkpoint needed."""
    catalog = pd.read_parquet(REPO_ROOT / "data" / brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)

    index_dir = REPO_ROOT / "indices" / brand / "active.faiss"
    faiss_index, article_ids = _load_faiss(index_dir)
    aid_to_row: dict[int, int] = {aid: i for i, aid in enumerate(article_ids)}

    return catalog, faiss_index, article_ids, aid_to_row


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _stratified_sample(catalog: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows spread across categories.

    Algorithm: allocate min-1 slot per category, then distribute remaining n
    slots proportionally by category size, capping at available items.
    """
    rng = np.random.default_rng(seed)
    cat_counts = catalog["category"].value_counts()
    n_cats = len(cat_counts)

    if n_cats >= n:
        # More categories than query budget: one item from each top-n category.
        selected = []
        for cat in cat_counts.head(n).index:
            rows = catalog[catalog["category"] == cat]
            selected.append(rows.sample(1, random_state=int(rng.integers(1_000_000))))
        return pd.concat(selected).reset_index(drop=True)

    # Allocate base = 1 per category, then distribute remaining proportionally.
    slots: dict[str, int] = {cat: 1 for cat in cat_counts.index}
    remaining = n - n_cats
    if remaining > 0:
        proportional = (cat_counts / cat_counts.sum() * remaining).round().astype(int)
        for cat, extra in proportional.items():
            max_extra = len(catalog[catalog["category"] == cat]) - slots[cat]
            slots[cat] += min(int(extra), max_extra)

    # Trim to exactly n (rounding may overshoot).
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
# Retrieval (mirrors /similar route in app/api/routes.py)
# ---------------------------------------------------------------------------


def _retrieve_similar(
    query_aid: int,
    faiss_index: faiss.Index,
    article_ids: list[int],
    aid_to_row: dict[int, int],
    k: int,
) -> list[tuple[int, float]]:
    row = aid_to_row.get(query_aid)
    if row is None:
        return []
    emb = faiss_index.reconstruct(row).reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(emb, k + 1)
    results = [
        (article_ids[idx], float(scores[0][j]))
        for j, idx in enumerate(indices[0])
        if idx != -1 and article_ids[idx] != query_aid
    ]
    return results[:k]


# ---------------------------------------------------------------------------
# Per-brand evaluation
# ---------------------------------------------------------------------------


def eval_brand(brand: str, n_queries: int = 25, k: int = 5, seed: int = 42) -> list[QueryResult]:
    print(f"  Loading {brand}...", end=" ", flush=True)
    catalog, faiss_index, article_ids, aid_to_row = load_brand(brand)
    art_map: dict[int, dict] = catalog.set_index("article_id").to_dict("index")
    print(f"{len(catalog)} items, {catalog['category'].nunique()} categories")

    queries = _stratified_sample(catalog, n=n_queries, seed=seed)
    print(f"  Sampled {len(queries)} queries across {queries['category'].nunique()} categories")

    results: list[QueryResult] = []
    for _, qrow in queries.iterrows():
        q_aid = int(qrow["article_id"])
        q_cat = str(qrow["category"])
        q_price = float(qrow["price_inr"]) if pd.notna(qrow["price_inr"]) else 0.0

        raw = _retrieve_similar(q_aid, faiss_index, article_ids, aid_to_row, k=k)

        neighbors: list[NeighborResult] = []
        for n_aid, score in raw:
            meta = art_map.get(n_aid, {})
            n_cat = str(meta.get("category", "unknown"))
            n_price_raw = meta.get("price_inr")
            n_price = (
                float(n_price_raw) if n_price_raw is not None and pd.notna(n_price_raw) else 0.0
            )
            is_self = n_aid == q_aid
            neighbors.append(NeighborResult(
                article_id=n_aid,
                title=str(meta.get("title", str(n_aid))),
                category=n_cat,
                price_inr=n_price,
                image_url=str(meta.get("image_url", "")),
                pdp_url=str(meta.get("pdp_url", "")),
                score=score,
                is_category_match=(n_cat == q_cat),
                is_self=is_self,
                is_near_dupe=(score >= NEAR_DUPE_THRESHOLD and not is_self),
            ))

        cat_matches = sum(1 for nb in neighbors if nb.is_category_match and not nb.is_self)
        n_valid = sum(1 for nb in neighbors if not nb.is_self)
        cat_match_rate = cat_matches / n_valid if n_valid > 0 else 0.0

        price_deltas = [
            abs(q_price - nb.price_inr)
            for nb in neighbors
            if not nb.is_self and nb.price_inr > 0 and q_price > 0
        ]
        mean_delta = float(np.mean(price_deltas)) if price_deltas else math.nan

        results.append(QueryResult(
            brand=brand,
            query_article_id=q_aid,
            query_title=str(qrow["title"]),
            query_category=q_cat,
            query_price_inr=q_price,
            query_image_url=str(qrow.get("image_url", "")),
            query_pdp_url=str(qrow.get("pdp_url", "")),
            neighbors=neighbors,
            category_match_rate=cat_match_rate,
            mean_price_delta_inr=mean_delta,
            has_self_match=any(nb.is_self for nb in neighbors),
            n_near_dupes=sum(1 for nb in neighbors if nb.is_near_dupe),
        ))

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _brand_stats(results: list[QueryResult]) -> dict:
    cat_rates = [r.category_match_rate for r in results]
    deltas = [r.mean_price_delta_inr for r in results if not math.isnan(r.mean_price_delta_inr)]

    per_cat: dict[str, list[float]] = {}
    for r in results:
        per_cat.setdefault(r.query_category, []).append(r.category_match_rate)

    return {
        "n_queries": len(results),
        "cat_match_rate": float(np.mean(cat_rates)),
        "cat_match_std": float(np.std(cat_rates)),
        "mean_price_delta": float(np.mean(deltas)) if deltas else math.nan,
        "self_match_count": sum(1 for r in results if r.has_self_match),
        "near_dupe_count": sum(r.n_near_dupes for r in results),
        "per_category": {
            cat: {
                "n": len(vals),
                "mean_match_rate": float(np.mean(vals)),
            }
            for cat, vals in per_cat.items()
        },
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f5f5f5; color: #222; font-size: 14px; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 28px 0 12px; border-left: 4px solid #4F46E5;
     padding-left: 10px; }
h3 { font-size: 14px; font-weight: 600; margin: 16px 0 8px; color: #555; }
.meta { color: #777; font-size: 12px; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
th { background: #1e1e2e; color: #fff; padding: 8px 12px; text-align: left; font-size: 12px; }
td { padding: 7px 12px; border-bottom: 1px solid #e5e5e5; font-size: 12px; }
tr:hover td { background: #fafafa; }
.bar-cell { position: relative; }
.bar { display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }
.bar-bg { display: inline-block; width: 80px; background: #e0e0e0; height: 10px;
          border-radius: 3px; vertical-align: middle; overflow: hidden; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px;
         font-weight: 600; }
.badge-good { background: #d1fae5; color: #065f46; }
.badge-ok { background: #fef3c7; color: #92400e; }
.badge-bad { background: #fee2e2; color: #991b1b; }
.badge-info { background: #dbeafe; color: #1e40af; }
.badge-dupe { background: #f3e8ff; color: #6b21a8; }
.badge-self { background: #ffe4e6; color: #9f1239; }
.callout { background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px;
           padding: 14px 18px; margin: 16px 0; }
.callout-warn { background: #fff1f2; border-color: #fecdd3; }
.callout-good { background: #f0fdf4; border-color: #bbf7d0; }
.callout h3 { margin-top: 0; color: #92400e; }
.callout-warn h3 { color: #be123c; }
.callout-good h3 { color: #166534; }
/* Query card layout */
.query-row { display: flex; gap: 12px; margin-bottom: 16px; background: #fff;
             border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px;
             align-items: flex-start; }
.query-row.worst { border-color: #fca5a5; background: #fff5f5; }
.item-card { width: 150px; flex-shrink: 0; text-align: center; }
.item-card img { width: 140px; height: 160px; object-fit: cover; border-radius: 4px;
                 background: #eee; }
.item-card .item-title { font-size: 11px; margin-top: 4px; line-height: 1.3;
                         overflow: hidden; display: -webkit-box;
                         -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.item-card .item-cat { font-size: 10px; margin-top: 2px; }
.item-card .item-price { font-size: 11px; font-weight: 600; color: #4F46E5; margin-top: 2px; }
.item-card .item-score { font-size: 10px; color: #777; }
.arrow { align-self: center; font-size: 22px; color: #aaa; flex-shrink: 0; }
.neighbors { display: flex; gap: 8px; flex-wrap: wrap; }
.query-label { font-size: 10px; font-weight: 700; text-transform: uppercase;
               color: #fff; background: #4F46E5; padding: 2px 6px; border-radius: 3px;
               display: inline-block; margin-bottom: 4px; }
.neighbor-label { font-size: 10px; color: #777; margin-bottom: 2px; }
.worst-brand { font-size: 11px; color: #555; }
.section-divider { border: none; border-top: 2px solid #e0e0e0; margin: 28px 0; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.stat-box { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; }
.stat-box .stat-val { font-size: 24px; font-weight: 700; color: #4F46E5; }
.stat-box .stat-label { font-size: 11px; color: #777; margin-top: 2px; }
.toc { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px 18px;
       margin-bottom: 20px; }
.toc a { color: #4F46E5; text-decoration: none; font-size: 13px; }
.toc a:hover { text-decoration: underline; }
.toc li { margin: 4px 0; }
"""


def _esc(s: str) -> str:
    return html_lib.escape(str(s))


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _inr(v: float) -> str:
    if math.isnan(v):
        return "—"
    return f"₹{v:,.0f}"


def _match_badge(rate: float) -> str:
    if rate >= 0.8:
        cls, label = "badge-good", f"{_pct(rate)} ✓"
    elif rate >= 0.4:
        cls, label = "badge-ok", f"{_pct(rate)} ~"
    else:
        cls, label = "badge-bad", f"{_pct(rate)} ✗"
    return f'<span class="badge {cls}">{label}</span>'


def _bar_html(rate: float) -> str:
    width = int(rate * 80)
    if rate >= 0.8:
        color = "#10b981"
    elif rate >= 0.4:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<div class="bar-bg"><div class="bar" style="width:{width}px;background:{color}"></div>'
        f"</div>"
    )


def _item_card_html(
    title: str,
    category: str,
    price_inr: float,
    image_url: str,
    pdp_url: str,
    *,
    query_category: str = "",
    score: float | None = None,
    is_query: bool = False,
    is_near_dupe: bool = False,
    rank: int | None = None,
) -> str:
    img_html = (
        f'<a href="{_esc(pdp_url)}" target="_blank">'
        f'<img src="{_esc(image_url)}" alt="{_esc(title)}" loading="lazy" '
        f'onerror="this.style.background=\'#ddd\';this.removeAttribute(\'src\')"></a>'
    )
    if is_query:
        label = '<div class="query-label">Query</div>'
    else:
        rank_str = f"#{rank} " if rank else ""
        is_match = category == query_category
        cat_cls = "badge-good" if is_match else "badge-bad"
        dupe_badge = ' <span class="badge badge-dupe">dupe?</span>' if is_near_dupe else ""
        label = (
            f'<div class="neighbor-label">{rank_str}'
            f'<span class="badge {cat_cls}">{_esc(category)}</span>{dupe_badge}</div>'
        )
    score_html = (
        f'<div class="item-score">sim {score:.3f}</div>' if score is not None else ""
    )
    return (
        f'<div class="item-card">'
        f"{label}"
        f"{img_html}"
        f'<div class="item-title">{_esc(title)}</div>'
        f'<div class="item-price">{_inr(price_inr)}</div>'
        f"{score_html}"
        f"</div>"
    )


def _query_row_html(r: QueryResult, *, is_worst: bool = False, rank_label: str = "") -> str:
    row_cls = "query-row worst" if is_worst else "query-row"
    brand_badge = f'<span class="badge badge-info" style="margin-right:4px">{_esc(r.brand)}</span>'
    rate_badge = _match_badge(r.category_match_rate)
    delta_str = _inr(r.mean_price_delta_inr)
    self_badge = ' — <span class="badge badge-self">self?</span>' if r.has_self_match else ""
    header = (
        f'<div style="margin-bottom:8px;font-size:12px">'
        f"{brand_badge}"
        f'<strong>{_esc(r.query_category)}</strong>'
        f" — cat match: {rate_badge}"
        f" — |ΔPrice|: {delta_str}"
        f"{self_badge}"
        f"</div>"
    )
    query_card = _item_card_html(
        r.query_title, r.query_category, r.query_price_inr,
        r.query_image_url, r.query_pdp_url, is_query=True,
    )
    neighbor_cards = "".join(
        _item_card_html(
            nb.title, nb.category, nb.price_inr, nb.image_url, nb.pdp_url,
            query_category=r.query_category, score=nb.score,
            is_near_dupe=nb.is_near_dupe, rank=i + 1,
        )
        for i, nb in enumerate(r.neighbors)
    )
    return (
        f'<div class="{row_cls}">'
        f"<div>"
        f"{header}"
        f'<div class="neighbors">'
        f"{query_card}"
        f'<div class="arrow">→</div>'
        f'<div class="neighbors">{neighbor_cards}</div>'
        f"</div>"
        f"</div>"
        f"</div>"
    )


def _summary_table_html(all_results: list[QueryResult], stats: dict[str, dict]) -> str:
    rows_html = ""
    for brand in BRANDS:
        if brand not in stats:
            continue
        s = stats[brand]
        rate = s["cat_match_rate"]
        rows_html += (
            f"<tr>"
            f"<td><strong>{_esc(brand)}</strong></td>"
            f"<td>{s['n_queries']}</td>"
            f"<td>{_bar_html(rate)} {_match_badge(rate)}</td>"
            f"<td>{_inr(s['mean_price_delta'])}</td>"
            f"<td>{'⚠ ' + str(s['self_match_count']) if s['self_match_count'] > 0 else '✓ 0'}</td>"
            f"<td>{s['near_dupe_count']}</td>"
            f"</tr>"
        )
    return (
        f"<table>"
        f"<tr><th>Brand</th><th>Queries</th><th>Cat Match Rate</th>"
        f"<th>Mean |ΔPrice|</th><th>Self-Match</th><th>Near Dupes (sim≥0.995)</th></tr>"
        f"{rows_html}"
        f"</table>"
    )


def _per_category_table_html(cat_data: dict[str, dict]) -> str:
    sorted_cats = sorted(cat_data.items(), key=lambda x: x[1]["mean_match_rate"])
    rows_html = "".join(
        f"<tr>"
        f"<td>{_esc(cat)}</td>"
        f"<td>{d['n']}</td>"
        f"<td>{_bar_html(d['mean_match_rate'])} {_match_badge(d['mean_match_rate'])}</td>"
        f"</tr>"
        for cat, d in sorted_cats
    )
    return (
        f"<table>"
        f"<tr><th>Category</th><th>Queries</th><th>Avg Cat Match Rate</th></tr>"
        f"{rows_html}"
        f"</table>"
    )


def _fashor_callout_html(fashor_stats: dict) -> str:
    rate = fashor_stats["cat_match_rate"]
    per_cat = fashor_stats["per_category"]

    # Compare ethnic vs western-cut categories
    ethnic_cats = {k: v for k, v in per_cat.items() if any(
        kw in k.lower() for kw in ["kurta", "ethnic", "lehenga", "dupatta", "anarkali"]
    )}
    western_cats = {k: v for k, v in per_cat.items() if any(
        kw in k.lower() for kw in ["dress", "top", "jeans", "co-ord", "fashion"]
    )}

    ethnic_rate = (
        float(np.mean([v["mean_match_rate"] for v in ethnic_cats.values()]))
        if ethnic_cats else math.nan
    )
    western_rate = (
        float(np.mean([v["mean_match_rate"] for v in western_cats.values()]))
        if western_cats else math.nan
    )

    cls = "callout-warn" if rate < 0.5 else ("callout" if rate < 0.8 else "callout-good")
    verdict = (
        "⚠ Low" if rate < 0.5
        else ("~ Moderate" if rate < 0.8 else "✓ Strong")
    )

    ethnic_str = _pct(ethnic_rate) if not math.isnan(ethnic_rate) else "n/a"
    western_str = _pct(western_rate) if not math.isnan(western_rate) else "n/a"

    ethnic_rows = "".join(
        f"<li>{_esc(k)}: {_pct(v['mean_match_rate'])} ({v['n']} queries)</li>"
        for k, v in sorted(ethnic_cats.items(), key=lambda x: x[1]["mean_match_rate"])
    ) or "<li>(no ethnic categories matched keywords)</li>"

    return (
        f'<div class="callout {cls}">'
        f"<h3>Fashor Ethnic Wear Analysis (H&M-trained tower)</h3>"
        f"<p><strong>Overall cat match rate:</strong> {_pct(rate)} — {verdict}</p>"
        f"<p style='margin-top:6px'><strong>Ethnic avg:</strong> {ethnic_str}"
        f" &nbsp;|&nbsp; <strong>Western avg:</strong> {western_str}</p>"
        f"<p style='margin-top:8px;font-size:12px;color:#555'>"
        f"The item tower (CLIP ViT-B/32 + SBERT) was trained on H&M western-fashion interactions. "
        f"Kurta/ethnic categories are out-of-distribution for the user tower but CLIP's visual "
        f"embeddings still capture garment shape, color, and texture — so similarity within ethnic "
        f"categories depends entirely on visual+text alignment, not interaction-based fine-tuning."
        f"</p>"
        f"<ul style='margin-top:8px;font-size:12px;padding-left:16px'>{ethnic_rows}</ul>"
        f"</div>"
    )


def build_html(all_results: list[QueryResult], n_queries: int = 25, k: int = 5) -> str:
    stats: dict[str, dict] = {
        brand: _brand_stats([r for r in all_results if r.brand == brand])
        for brand in BRANDS
        if any(r.brand == brand for r in all_results)
    }

    worst_10 = sorted(all_results, key=lambda r: r.category_match_rate)[:10]
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Table of contents
    toc_html = (
        '<div class="toc">'
        "<strong>Jump to:</strong>"
        "<ul>"
        '<li><a href="#summary">Headline Summary</a></li>'
        '<li><a href="#worst10">Worst 10 Results</a></li>'
        + "".join(f'<li><a href="#{brand}">{brand.title()}</a></li>' for brand in BRANDS)
        + "</ul></div>"
    )

    # Headline stats
    all_rates = [r.category_match_rate for r in all_results]
    overall_rate = float(np.mean(all_rates)) if all_rates else math.nan
    all_deltas = [
        r.mean_price_delta_inr for r in all_results if not math.isnan(r.mean_price_delta_inr)
    ]
    overall_delta = float(np.mean(all_deltas)) if all_deltas else math.nan
    total_self = sum(1 for r in all_results if r.has_self_match)
    total_dupes = sum(r.n_near_dupes for r in all_results)

    stat_grid_html = (
        '<div class="stat-grid">'
        f'<div class="stat-box"><div class="stat-val">{_pct(overall_rate)}</div>'
        f'<div class="stat-label">Overall cat match rate ({len(all_results)} queries)</div></div>'
        f'<div class="stat-box"><div class="stat-val">{_inr(overall_delta)}</div>'
        f'<div class="stat-label">Mean |ΔPrice| across all queries</div></div>'
        f'<div class="stat-box"><div class="stat-val">'
        f'{"⚠ " + str(total_self) if total_self else "✓ 0"}</div>'
        f'<div class="stat-label">Self-match violations</div></div>'
        f'<div class="stat-box"><div class="stat-val">{total_dupes}</div>'
        f'<div class="stat-label">Near-dupe flags (sim≥0.995)</div></div>'
        "</div>"
    )

    # Per-brand sections
    brand_sections_html = ""
    for brand in BRANDS:
        brand_results = [r for r in all_results if r.brand == brand]
        if not brand_results:
            continue
        s = stats[brand]
        callout = _fashor_callout_html(s) if brand == "fashor" else ""
        cat_table = _per_category_table_html(s["per_category"])
        query_cards = "".join(_query_row_html(r) for r in brand_results)
        brand_sections_html += (
            f'<hr class="section-divider">'
            f'<h2 id="{_esc(brand)}">{_esc(brand.title())} — '
            f'cat match {_match_badge(s["cat_match_rate"])} | '
            f'{_inr(s["mean_price_delta"])} avg |ΔPrice|</h2>'
            f"{callout}"
            f"<h3>Per-Category Breakdown</h3>"
            f"{cat_table}"
            f"<h3>Query ↦ Top-{k} Neighbors</h3>"
            f"{query_cards}"
        )

    # Worst 10
    worst_html = "".join(
        _query_row_html(r, is_worst=True)
        for r in worst_10
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Similarity Quality Eval — Fashion Recommender</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>Similarity Quality Eval — Fashion Recommender</h1>
  <p class="meta">Generated {_esc(date_str)} &nbsp;·&nbsp; {n_queries} queries/brand &nbsp;·&nbsp;
     top-{k} neighbors &nbsp;·&nbsp; brands: snitch, fashor, powerlook</p>

  {toc_html}

  <h2 id="summary">Headline Summary</h2>
  {stat_grid_html}
  {_summary_table_html(all_results, stats)}

  <h2 id="worst10">⚠ Worst 10 Results (Lowest Category-Match)</h2>
  <p class="meta">These are the 10 query items across all brands where /similar returned the
     fewest on-category neighbors — the primary failure-mode candidates for pre-sale inspection.</p>
  {worst_html}

  {brand_sections_html}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Similarity-quality inspection report for Indian brand catalogs."
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output HTML path (default: reports/similarity_eval.html)")
    p.add_argument("--n-queries", type=int, default=25, metavar="N",
                   help="Query items per brand (default: 25)")
    p.add_argument("--k", type=int, default=5, metavar="K",
                   help="Neighbors per query (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    # Ensure UTF-8 output on Windows (rupee sign ₹ not in cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Similarity Quality Eval — {len(BRANDS)} brands, "
          f"{args.n_queries} queries each, top-{args.k} neighbors")
    print(f"Output: {args.output}\n")

    all_results: list[QueryResult] = []
    for brand in BRANDS:
        print(f"[{brand}]")
        results = eval_brand(brand, n_queries=args.n_queries, k=args.k, seed=args.seed)
        all_results.extend(results)

        # Headline per brand
        s = _brand_stats(results)
        print(f"  cat_match_rate={s['cat_match_rate']:.3f}  "
              f"mean_dPrice={_inr(s['mean_price_delta'])}  "
              f"self_match={s['self_match_count']}  "
              f"near_dupes={s['near_dupe_count']}")
        print()

    # Worst 10 summary
    worst_10 = sorted(all_results, key=lambda r: r.category_match_rate)[:10]
    print("=== Worst 10 queries (lowest cat match rate) ===")
    for i, r in enumerate(worst_10, 1):
        print(f"  {i:2d}. {r.brand:10s} {r.query_category:30s} "
              f"match={_pct(r.category_match_rate)}  aid={r.query_article_id}")

    print()
    html_content = build_html(all_results, n_queries=args.n_queries, k=args.k)
    args.output.write_text(html_content, encoding="utf-8")
    print(f"Report written to: {args.output}")
    print(f"Open with: start {args.output}")


if __name__ == "__main__":
    main()
