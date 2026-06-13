"""
scripts/eval_similarity_quality.py

Before/after similarity-quality inspection for Indian brand catalogs.
Compares raw FAISS retrieval against the re-rank layer (app.rerank) for /similar.

For each brand (snitch, fashor, powerlook):
  1. Loads per-brand RerankConfig from brands/{brand}.yaml.
  2. Samples N query items spread across categories (stratified, seed=42).
  3. Runs TWO retrieval passes per query:
       raw      — pure FAISS top-k (mirrors /similar with rerank disabled)
       reranked — FAISS pool → rerank() (mirrors /similar with rerank enabled)
  4. Computes TWO category-match metrics per pass:
       strict   — exact category label match only (original metric; comparable to any
                  baseline — not inflated by our taxonomy choices)
       affinity — affinity(query_cat, neighbor_cat) >= AFFINITY_THRESHOLD (0.4).
                  Counts exact (1.0) + equivalent-group (0.7) + related-group (0.4).
                  Threshold is stated explicitly so taxonomy-driven improvement is
                  auditable against the strict number.
  5. Reports mean |ΔPrice| (INR) before/after rerank.
  6. Checks per-brand guardrails (Fashor strict ≥ 58%, Powerlook strict ≥ 64%,
     Fashor |ΔPrice| must drop; Snitch has no hard floor).
  7. Dumps a browsable HTML report.

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
from typing import Any

import faiss
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rerank import CategoryAffinityMap, RerankConfig
from app.rerank import rerank as _rerank_fn

BRANDS: list[str] = ["snitch", "fashor", "powerlook"]
NEAR_DUPE_THRESHOLD = 0.995
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "similarity_eval.html"

# Affinity match threshold = related_group_bonus (0.4).
# Counts exact (1.0) + equivalent-group (0.7) + related-group (0.4) as a match.
# Stated explicitly so affinity improvement vs. taxonomy choice is auditable.
AFFINITY_THRESHOLD = 0.4

# Per-brand guardrails checked against the RERANKED pass.
# strict_floor: minimum reranked strict cat-match rate (None = no hard floor).
# price_must_drop: reranked mean |ΔPrice| must be strictly lower than raw.
GUARDRAILS: dict[str, dict[str, Any]] = {
    "fashor": {
        "strict_floor": 0.58,
        "price_must_drop": True,
        "note": "price weight=0.25 (highest); if price drops but strict regresses, flag weight as too high",
    },
    "powerlook": {
        "strict_floor": 0.64,
        "price_must_drop": False,
        "note": "strongest baseline; protect strict floor",
    },
    "snitch": {
        "strict_floor": None,
        "price_must_drop": False,
        "note": "no hard floor; report before/after only",
    },
}


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
    is_strict_match: bool    # n_category == query_category (exact label only)
    is_affinity_match: bool  # affinity(query_cat, n_cat) >= AFFINITY_THRESHOLD
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
    raw_neighbors: list[NeighborResult]
    reranked_neighbors: list[NeighborResult]
    raw_strict_rate: float
    raw_affinity_rate: float
    raw_mean_price_delta: float
    reranked_strict_rate: float
    reranked_affinity_rate: float
    reranked_mean_price_delta: float
    has_self_match: bool    # reranked pass
    n_near_dupes: int       # reranked pass
    # Diversity metrics (Feature 1 / Feature 2 eval)
    raw_inter_dupe_pairs: int       # unordered pairs in raw top-k with cosine >= dupe_sim_threshold
    rkd_inter_dupe_pairs: int       # same, reranked top-k
    raw_distinct_categories: int    # distinct categories in raw top-k
    rkd_distinct_categories: int    # distinct categories in reranked top-k
    raw_same_band_rate: float   # fraction of raw neighbors in query's price band (nan if no bands)
    rkd_same_band_rate: float   # fraction of reranked neighbors in query's price band


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_faiss(index_dir: Path) -> tuple[faiss.Index, list[int]]:
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "article_ids.pkl", "rb") as fh:
        raw_ids = pickle.load(fh)  # noqa: S301
    return index, [int(aid) for aid in raw_ids]


def _load_rerank_config(brand: str) -> RerankConfig:
    yaml_path = REPO_ROOT / "brands" / f"{brand}.yaml"
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    return RerankConfig.model_validate(data.get("rerank", {}))


def load_brand(
    brand: str,
) -> tuple[pd.DataFrame, faiss.Index, list[int], dict[int, int], RerankConfig]:
    catalog = pd.read_parquet(REPO_ROOT / "data" / brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)
    index_dir = REPO_ROOT / "indices" / brand / "active.faiss"
    faiss_index, article_ids = _load_faiss(index_dir)
    aid_to_row: dict[int, int] = {aid: i for i, aid in enumerate(article_ids)}
    rerank_config = _load_rerank_config(brand)
    return catalog, faiss_index, article_ids, aid_to_row, rerank_config


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _stratified_sample(catalog: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows spread across categories (min-1 per category, proportional fill)."""
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
# Retrieval
# ---------------------------------------------------------------------------


def _build_neighbor(
    n_aid: int,
    score: float,
    art_map: dict[int, dict],
    query_aid: int,
    query_category: str,
    query_price: float,
    affinity_map: CategoryAffinityMap,
) -> NeighborResult:
    meta = art_map.get(n_aid, {})
    n_cat = str(meta.get("category", "unknown"))
    n_price_raw = meta.get("price_inr")
    try:
        n_price = float(n_price_raw) if n_price_raw is not None else 0.0
    except (TypeError, ValueError):
        n_price = 0.0
    is_self = n_aid == query_aid
    aff = affinity_map.affinity(query_category, n_cat)
    return NeighborResult(
        article_id=n_aid,
        title=str(meta.get("title", str(n_aid))),
        category=n_cat,
        price_inr=n_price,
        image_url=str(meta.get("image_url", "")),
        pdp_url=str(meta.get("pdp_url", "")),
        score=score,
        is_strict_match=(n_cat == query_category),
        is_affinity_match=(aff >= AFFINITY_THRESHOLD),
        is_self=is_self,
        is_near_dupe=(score >= NEAR_DUPE_THRESHOLD and not is_self),
    )


def _retrieve_raw(
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
    return [
        (article_ids[idx], float(scores[0][j]))
        for j, idx in enumerate(indices[0])
        if idx != -1 and article_ids[idx] != query_aid
    ][:k]


def _retrieve_reranked(
    query_aid: int,
    faiss_index: faiss.Index,
    article_ids: list[int],
    aid_to_row: dict[int, int],
    art_map: dict[int, dict],
    rerank_config: RerankConfig,
    k: int,
) -> list[tuple[int, float]]:
    row = aid_to_row.get(query_aid)
    if row is None:
        return []
    pool_k = rerank_config.candidate_pool_size
    emb = faiss_index.reconstruct(row).reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(emb, pool_k + 1)
    candidates = [
        (article_ids[idx], float(scores[0][j]))
        for j, idx in enumerate(indices[0])
        if idx != -1 and article_ids[idx] != query_aid
    ]
    query_meta = art_map.get(query_aid, {})
    query_price = float(query_meta.get("price_inr") or 0.0)
    query_cat = str(query_meta.get("category", ""))

    # Build candidate embeddings from the same FAISS index for MMR diversity (Feature 1).
    # Keyed by the same int art_id used in candidates so rerank() key normalisation is consistent.
    embeddings: dict[int, np.ndarray] | None = None
    if rerank_config.w_diversity > 0.0:
        embeddings = {
            aid: faiss_index.reconstruct(aid_to_row[aid])
            for aid, _ in candidates
            if aid in aid_to_row
        }

    return _rerank_fn(candidates, query_price, query_cat, art_map, rerank_config, k,
                      embeddings=embeddings)


# ---------------------------------------------------------------------------
# Per-brand evaluation
# ---------------------------------------------------------------------------


def _inter_dupe_pairs(
    neighbor_aids: list[int],
    faiss_index: faiss.Index,
    aid_to_row: dict[int, int],
    dupe_sim_threshold: float,
) -> int:
    """Count unordered pairs among neighbor_aids with pairwise cosine >= dupe_sim_threshold.

    Embeddings are reconstructed from the FAISS index (L2-normalised, so cosine = dot product).
    Pairs where either item is missing from the index are skipped.
    """
    vecs: list[np.ndarray] = []
    for aid in neighbor_aids:
        row = aid_to_row.get(aid)
        if row is not None:
            vecs.append(faiss_index.reconstruct(row).astype(np.float32))

    n = len(vecs)
    if n < 2:
        return 0

    matrix = np.stack(vecs)  # (n, dim)
    # Dot product matrix = cosine matrix for L2-normalised vectors
    cos_matrix = matrix @ matrix.T  # (n, n)

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cos_matrix[i, j] >= dupe_sim_threshold:
                count += 1
    return count


def _neighbor_metrics(
    neighbors: list[NeighborResult],
    query_price: float,
) -> tuple[float, float, float]:
    """Returns (strict_rate, affinity_rate, mean_price_delta)."""
    valid = [nb for nb in neighbors if not nb.is_self]
    n_valid = len(valid)
    strict_rate = sum(1 for nb in valid if nb.is_strict_match) / n_valid if n_valid else 0.0
    affinity_rate = sum(1 for nb in valid if nb.is_affinity_match) / n_valid if n_valid else 0.0
    deltas = [abs(query_price - nb.price_inr) for nb in valid if nb.price_inr > 0 and query_price > 0]
    mean_delta = float(np.mean(deltas)) if deltas else math.nan
    return strict_rate, affinity_rate, mean_delta


def _diversity_metrics(
    neighbors: list[NeighborResult],
    query_price: float,
    rerank_config: RerankConfig,
    faiss_index: faiss.Index,
    aid_to_row: dict[int, int],
) -> tuple[int, int, float]:
    """Compute diversity/coherence metrics for a list of neighbors.

    Returns:
        inter_dupe_pairs: unordered pairs with cosine >= dupe_sim_threshold
        distinct_categories: count of distinct category labels in top-k
        same_band_rate: fraction of neighbors in query's price band (math.nan if no bands)
    """
    valid = [nb for nb in neighbors if not nb.is_self]
    aids = [nb.article_id for nb in valid]

    inter_dupe = _inter_dupe_pairs(aids, faiss_index, aid_to_row, rerank_config.dupe_sim_threshold)

    distinct_cats = len({nb.category for nb in valid})

    if rerank_config.price_bands_inr and query_price > 0:
        from app.rerank import _price_band_index  # local import; already loaded via rerank import
        query_band = _price_band_index(query_price, rerank_config.price_bands_inr)
        in_band = sum(
            1 for nb in valid
            if nb.price_inr > 0
            and _price_band_index(nb.price_inr, rerank_config.price_bands_inr) == query_band
        )
        same_band_rate = in_band / len(valid) if valid else math.nan
    else:
        same_band_rate = math.nan

    return inter_dupe, distinct_cats, same_band_rate


def eval_brand(brand: str, n_queries: int = 25, k: int = 5, seed: int = 42) -> list[QueryResult]:
    print(f"  Loading {brand}...", end=" ", flush=True)
    catalog, faiss_index, article_ids, aid_to_row, rerank_config = load_brand(brand)
    art_map: dict[int, dict] = catalog.set_index("article_id").to_dict("index")
    print(f"{len(catalog)} items, {catalog['category'].nunique()} categories")

    queries = _stratified_sample(catalog, n=n_queries, seed=seed)
    print(f"  Sampled {len(queries)} queries across {queries['category'].nunique()} categories")

    affinity_map = CategoryAffinityMap(rerank_config)
    results: list[QueryResult] = []

    for _, qrow in queries.iterrows():
        q_aid = int(qrow["article_id"])
        q_cat = str(qrow["category"])
        q_price = float(qrow["price_inr"]) if pd.notna(qrow["price_inr"]) else 0.0

        raw_hits = _retrieve_raw(q_aid, faiss_index, article_ids, aid_to_row, k=k)
        rkd_hits = _retrieve_reranked(
            q_aid, faiss_index, article_ids, aid_to_row, art_map, rerank_config, k=k
        )

        raw_nbs = [
            _build_neighbor(aid, sc, art_map, q_aid, q_cat, q_price, affinity_map)
            for aid, sc in raw_hits
        ]
        rkd_nbs = [
            _build_neighbor(aid, sc, art_map, q_aid, q_cat, q_price, affinity_map)
            for aid, sc in rkd_hits
        ]

        raw_strict, raw_affinity, raw_delta = _neighbor_metrics(raw_nbs, q_price)
        rkd_strict, rkd_affinity, rkd_delta = _neighbor_metrics(rkd_nbs, q_price)

        raw_idp, raw_dcat, raw_sbr = _diversity_metrics(
            raw_nbs, q_price, rerank_config, faiss_index, aid_to_row
        )
        rkd_idp, rkd_dcat, rkd_sbr = _diversity_metrics(
            rkd_nbs, q_price, rerank_config, faiss_index, aid_to_row
        )

        results.append(QueryResult(
            brand=brand,
            query_article_id=q_aid,
            query_title=str(qrow["title"]),
            query_category=q_cat,
            query_price_inr=q_price,
            query_image_url=str(qrow.get("image_url") or ""),
            query_pdp_url=str(qrow.get("pdp_url") or ""),
            raw_neighbors=raw_nbs,
            reranked_neighbors=rkd_nbs,
            raw_strict_rate=raw_strict,
            raw_affinity_rate=raw_affinity,
            raw_mean_price_delta=raw_delta,
            reranked_strict_rate=rkd_strict,
            reranked_affinity_rate=rkd_affinity,
            reranked_mean_price_delta=rkd_delta,
            has_self_match=any(nb.is_self for nb in rkd_nbs),
            n_near_dupes=sum(1 for nb in rkd_nbs if nb.is_near_dupe),
            raw_inter_dupe_pairs=raw_idp,
            rkd_inter_dupe_pairs=rkd_idp,
            raw_distinct_categories=raw_dcat,
            rkd_distinct_categories=rkd_dcat,
            raw_same_band_rate=raw_sbr,
            rkd_same_band_rate=rkd_sbr,
        ))

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _brand_stats(results: list[QueryResult]) -> dict:
    raw_strict = [r.raw_strict_rate for r in results]
    raw_affinity = [r.raw_affinity_rate for r in results]
    raw_deltas = [r.raw_mean_price_delta for r in results if not math.isnan(r.raw_mean_price_delta)]
    rkd_strict = [r.reranked_strict_rate for r in results]
    rkd_affinity = [r.reranked_affinity_rate for r in results]
    rkd_deltas = [r.reranked_mean_price_delta for r in results if not math.isnan(r.reranked_mean_price_delta)]

    per_cat: dict[str, dict] = {}
    for r in results:
        cat = r.query_category
        if cat not in per_cat:
            per_cat[cat] = {
                "n": 0,
                "raw_strict": [],
                "rkd_strict": [],
                "raw_affinity": [],
                "rkd_affinity": [],
            }
        per_cat[cat]["n"] += 1
        per_cat[cat]["raw_strict"].append(r.raw_strict_rate)
        per_cat[cat]["rkd_strict"].append(r.reranked_strict_rate)
        per_cat[cat]["raw_affinity"].append(r.raw_affinity_rate)
        per_cat[cat]["rkd_affinity"].append(r.reranked_affinity_rate)

    # Diversity / coherence aggregates
    raw_sbr_vals = [r.raw_same_band_rate for r in results if not math.isnan(r.raw_same_band_rate)]
    rkd_sbr_vals = [r.rkd_same_band_rate for r in results if not math.isnan(r.rkd_same_band_rate)]

    return {
        "n_queries": len(results),
        "raw_strict_rate": float(np.mean(raw_strict)) if raw_strict else math.nan,
        "raw_affinity_rate": float(np.mean(raw_affinity)) if raw_affinity else math.nan,
        "raw_mean_price_delta": float(np.mean(raw_deltas)) if raw_deltas else math.nan,
        "rkd_strict_rate": float(np.mean(rkd_strict)) if rkd_strict else math.nan,
        "rkd_affinity_rate": float(np.mean(rkd_affinity)) if rkd_affinity else math.nan,
        "rkd_mean_price_delta": float(np.mean(rkd_deltas)) if rkd_deltas else math.nan,
        "self_match_count": sum(1 for r in results if r.has_self_match),
        "near_dupe_count": sum(r.n_near_dupes for r in results),
        # New diversity/coherence metrics (Feature 1 + 2)
        "raw_inter_dupe_pairs": sum(r.raw_inter_dupe_pairs for r in results),
        "rkd_inter_dupe_pairs": sum(r.rkd_inter_dupe_pairs for r in results),
        "raw_distinct_categories_mean": float(
            np.mean([r.raw_distinct_categories for r in results])
        ),
        "rkd_distinct_categories_mean": float(
            np.mean([r.rkd_distinct_categories for r in results])
        ),
        "raw_same_band_rate": float(np.mean(raw_sbr_vals)) if raw_sbr_vals else math.nan,
        "rkd_same_band_rate": float(np.mean(rkd_sbr_vals)) if rkd_sbr_vals else math.nan,
        "per_category": {
            cat: {
                "n": data["n"],
                "raw_strict_rate": float(np.mean(data["raw_strict"])),
                "rkd_strict_rate": float(np.mean(data["rkd_strict"])),
                "raw_affinity_rate": float(np.mean(data["raw_affinity"])),
                "rkd_affinity_rate": float(np.mean(data["rkd_affinity"])),
            }
            for cat, data in per_cat.items()
        },
    }


# ---------------------------------------------------------------------------
# Guardrail checks + CLI table
# ---------------------------------------------------------------------------


def _check_guardrails(brand: str, stats: dict) -> tuple[bool, list[str]]:
    """Returns (all_hard_guardrails_pass, list_of_breach_and_flag_messages)."""
    g = GUARDRAILS.get(brand, {})
    raw_strict = stats["raw_strict_rate"]
    rkd_strict = stats["rkd_strict_rate"]
    raw_delta = stats["raw_mean_price_delta"]
    rkd_delta = stats["rkd_mean_price_delta"]
    breaches: list[str] = []
    flags: list[str] = []

    strict_floor = g.get("strict_floor")
    if strict_floor is not None and not math.isnan(rkd_strict):
        if rkd_strict < strict_floor:
            breaches.append(
                f"strict cat-match {_pct(rkd_strict)} < floor {_pct(strict_floor)}"
            )

    if g.get("price_must_drop"):
        if not math.isnan(raw_delta) and not math.isnan(rkd_delta):
            if rkd_delta >= raw_delta:
                breaches.append(
                    f"|ΔPrice| did not drop ({_inr(raw_delta)} → {_inr(rkd_delta)})"
                )

    # Fashor-specific advisory: price improved but strict regressed — price weight likely too high
    if brand == "fashor":
        if (
            not math.isnan(raw_delta)
            and not math.isnan(rkd_delta)
            and not math.isnan(raw_strict)
            and not math.isnan(rkd_strict)
        ):
            if rkd_delta < raw_delta and rkd_strict < raw_strict:
                flags.append(
                    f"price improved ({_inr(raw_delta)} → {_inr(rkd_delta)}) "
                    f"but strict regressed ({_pct(raw_strict)} → {_pct(rkd_strict)}); "
                    f"Fashor price weight 0.25 may be too high — consider 0.20"
                )

    ok = len(breaches) == 0
    messages = [f"BREACH: {b}" for b in breaches] + [f"FLAG: {f}" for f in flags]
    return ok, messages


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _inr(v: float) -> str:
    if math.isnan(v):
        return "—"
    return f"₹{v:,.0f}"


def _delta_pp(before: float, after: float) -> str:
    if math.isnan(before) or math.isnan(after):
        return "—"
    diff = (after - before) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.0f}pp"


def _delta_inr(before: float, after: float) -> str:
    if math.isnan(before) or math.isnan(after):
        return "—"
    arrow = "↓" if after < before else ("↑" if after > before else "→")
    return f"{arrow}{abs(before - after):,.0f}"


def _fmt_sbr(v: float) -> str:
    """Format same-band rate or return 'n/a' when not configured."""
    return _pct(v) if not math.isnan(v) else "n/a"


def print_comparison_table(all_stats: dict[str, dict]) -> bool:
    """Print the before/after CLI table. Returns True if all hard guardrails pass."""
    col_brand = 12
    col_strict = 22
    col_affinity = 24
    col_price = 24
    col_guard = 10

    header = (
        f"{'Brand':<{col_brand}} | "
        f"{'Strict (raw → rkd)':<{col_strict}} | "
        f"{'Affinity (raw → rkd)':<{col_affinity}} | "
        f"{'|ΔPrice| (raw → rkd)':<{col_price}} | "
        f"Guardrail"
    )
    sep = "-" * len(header)

    print(f"\nAffinity threshold: {AFFINITY_THRESHOLD} "
          f"(counts exact + equivalent-group + related-group)\n")
    print(sep)
    print(header)
    print(sep)

    all_ok = True
    for brand in BRANDS:
        if brand not in all_stats:
            continue
        s = all_stats[brand]
        ok, messages = _check_guardrails(brand, s)
        if not ok:
            all_ok = False

        rs, ra = s["raw_strict_rate"], s["raw_affinity_rate"]
        ks, ka = s["rkd_strict_rate"], s["rkd_affinity_rate"]
        rd, kd = s["raw_mean_price_delta"], s["rkd_mean_price_delta"]

        strict_str = f"{_pct(rs)} → {_pct(ks)} ({_delta_pp(rs, ks)})"
        affinity_str = f"{_pct(ra)} → {_pct(ka)} ({_delta_pp(ra, ka)})"
        price_str = f"{_inr(rd)} → {_inr(kd)} ({_delta_inr(rd, kd)})"
        guard_str = "BREACH" if not ok else ("FLAG" if messages else "OK")

        print(
            f"{brand:<{col_brand}} | "
            f"{strict_str:<{col_strict}} | "
            f"{affinity_str:<{col_affinity}} | "
            f"{price_str:<{col_price}} | "
            f"{guard_str}"
        )
        for msg in messages:
            print(f"  {'':>{col_brand}}   {msg}")

        # Diversity / coherence sub-row (new Feature 1 + 2 metrics)
        raw_idp = s.get("raw_inter_dupe_pairs", 0)
        rkd_idp = s.get("rkd_inter_dupe_pairs", 0)
        raw_dc = s.get("raw_distinct_categories_mean", math.nan)
        rkd_dc = s.get("rkd_distinct_categories_mean", math.nan)
        raw_sbr = s.get("raw_same_band_rate", math.nan)
        rkd_sbr = s.get("rkd_same_band_rate", math.nan)
        dc_str = (
            f"{raw_dc:.1f} → {rkd_dc:.1f}"
            if not math.isnan(raw_dc) and not math.isnan(rkd_dc)
            else "—"
        )
        print(
            f"  {'':>{col_brand}}   "
            f"inter_dupe_pairs(raw={raw_idp} rkd={rkd_idp})  "
            f"distinct_cats(mean): {dc_str}  "
            f"same_band_rate(raw={_fmt_sbr(raw_sbr)} rkd={_fmt_sbr(rkd_sbr)})"
        )

    print(sep)
    return all_ok


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
.badge-breach { background: #fee2e2; color: #991b1b; font-size: 12px; padding: 3px 8px; }
.badge-flag { background: #fef3c7; color: #92400e; font-size: 12px; padding: 3px 8px; }
.badge-pass { background: #d1fae5; color: #065f46; font-size: 12px; padding: 3px 8px; }
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
.delta-good { color: #065f46; font-weight: 600; }
.delta-bad { color: #991b1b; font-weight: 600; }
.delta-neutral { color: #555; }
"""


def _esc(s: str) -> str:
    return html_lib.escape(str(s))


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
    color = "#10b981" if rate >= 0.8 else ("#f59e0b" if rate >= 0.4 else "#ef4444")
    return (
        f'<div class="bar-bg"><div class="bar" style="width:{width}px;background:{color}"></div>'
        f"</div>"
    )


def _delta_html(before: float, after: float, lower_is_better: bool = False) -> str:
    if math.isnan(before) or math.isnan(after):
        return '<span class="delta-neutral">—</span>'
    diff = after - before
    if lower_is_better:
        is_good = diff < 0
    else:
        is_good = diff > 0
    cls = "delta-good" if is_good else ("delta-bad" if diff != 0 else "delta-neutral")
    sign = "+" if diff >= 0 else ""
    if lower_is_better:
        arrow = "↓" if diff < 0 else ("↑" if diff > 0 else "→")
        return f'<span class="{cls}">{arrow}{abs(diff):,.0f}</span>'
    return f'<span class="{cls}">{sign}{diff * 100:.0f}pp</span>'


def _item_card_html(
    title: str,
    category: str,
    price_inr: float,
    image_url: str,
    pdp_url: str,
    *,
    query_category: str = "",
    score: float | None = None,
    is_strict_match: bool = False,
    is_affinity_match: bool = False,
    is_near_dupe: bool = False,
    is_query: bool = False,
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
        if is_strict_match:
            cat_cls = "badge-good"
            cat_note = ""
        elif is_affinity_match:
            cat_cls = "badge-ok"
            cat_note = " ~"  # affinity-only match
        else:
            cat_cls = "badge-bad"
            cat_note = ""
        dupe_badge = ' <span class="badge badge-dupe">dupe?</span>' if is_near_dupe else ""
        label = (
            f'<div class="neighbor-label">{rank_str}'
            f'<span class="badge {cat_cls}">{_esc(category)}{cat_note}</span>{dupe_badge}</div>'
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


def _query_row_html(r: QueryResult, *, is_worst: bool = False) -> str:
    """Render one query row showing RERANKED neighbors with strict + affinity badges."""
    row_cls = "query-row worst" if is_worst else "query-row"
    brand_badge = f'<span class="badge badge-info" style="margin-right:4px">{_esc(r.brand)}</span>'

    strict_delta = _delta_html(r.raw_strict_rate, r.reranked_strict_rate)
    affinity_delta = _delta_html(r.raw_affinity_rate, r.reranked_affinity_rate)
    price_delta = _delta_html(r.raw_mean_price_delta, r.reranked_mean_price_delta, lower_is_better=True)

    self_badge = ' <span class="badge badge-self">self?</span>' if r.has_self_match else ""
    header = (
        f'<div style="margin-bottom:8px;font-size:12px">'
        f"{brand_badge}"
        f'<strong>{_esc(r.query_category)}</strong>'
        f" — strict: {_match_badge(r.reranked_strict_rate)} ({strict_delta})"
        f" — affinity: {_match_badge(r.reranked_affinity_rate)} ({affinity_delta})"
        f" — |ΔPrice|: {_inr(r.reranked_mean_price_delta)} ({price_delta})"
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
            is_strict_match=nb.is_strict_match,
            is_affinity_match=nb.is_affinity_match,
            is_near_dupe=nb.is_near_dupe, rank=i + 1,
        )
        for i, nb in enumerate(r.reranked_neighbors)
    )
    return (
        f'<div class="{row_cls}">'
        f"<div style='width:100%'>"
        f"{header}"
        f'<div class="neighbors">'
        f"{query_card}"
        f'<div class="arrow">→</div>'
        f'<div class="neighbors">{neighbor_cards}</div>'
        f"</div>"
        f"</div>"
        f"</div>"
    )


def _comparison_table_html(all_stats: dict[str, dict]) -> str:
    """Before/after table: brand × strict/affinity/price (raw → reranked)."""
    rows_html = ""
    for brand in BRANDS:
        if brand not in all_stats:
            continue
        s = all_stats[brand]
        ok, messages = _check_guardrails(brand, s)
        guard_cls = "badge-pass" if ok and not messages else ("badge-flag" if ok else "badge-breach")
        guard_label = "OK" if ok and not messages else ("FLAG" if ok else "BREACH")

        rs, ra = s["raw_strict_rate"], s["raw_affinity_rate"]
        ks, ka = s["rkd_strict_rate"], s["rkd_affinity_rate"]
        rd, kd = s["raw_mean_price_delta"], s["rkd_mean_price_delta"]

        msg_html = "".join(
            f'<div style="font-size:11px;color:#555;margin-top:3px">{_esc(m)}</div>'
            for m in messages
        )

        rows_html += (
            f"<tr>"
            f"<td><strong>{_esc(brand)}</strong></td>"
            f"<td>{s['n_queries']}</td>"
            f"<td>{_bar_html(rs)} {_match_badge(rs)}</td>"
            f"<td>{_bar_html(ks)} {_match_badge(ks)} {_delta_html(rs, ks)}</td>"
            f"<td>{_bar_html(ra)} {_match_badge(ra)}</td>"
            f"<td>{_bar_html(ka)} {_match_badge(ka)} {_delta_html(ra, ka)}</td>"
            f"<td>{_inr(rd)}</td>"
            f"<td>{_inr(kd)} {_delta_html(rd, kd, lower_is_better=True)}</td>"
            f'<td><span class="badge {guard_cls}">{guard_label}</span>{msg_html}</td>'
            f"</tr>"
        )

    return (
        f"<table>"
        f"<tr>"
        f"<th>Brand</th><th>Queries</th>"
        f"<th>Strict (raw)</th><th>Strict (reranked)</th>"
        f"<th>Affinity (raw)</th><th>Affinity (reranked)</th>"
        f"<th>|ΔPrice| (raw)</th><th>|ΔPrice| (reranked)</th>"
        f"<th>Guardrail</th>"
        f"</tr>"
        f"{rows_html}"
        f"</table>"
        f'<p class="meta">Affinity threshold: {AFFINITY_THRESHOLD} — '
        f"counts exact (1.0) + equivalent-group (0.7) + related-group (0.4) as matches. "
        f"Green item-card badge = strict match; yellow ~ = affinity-only match; red = no match.</p>"
    )


def _per_category_table_html(cat_data: dict[str, dict]) -> str:
    sorted_cats = sorted(cat_data.items(), key=lambda x: x[1]["rkd_strict_rate"])
    rows_html = "".join(
        f"<tr>"
        f"<td>{_esc(cat)}</td>"
        f"<td>{d['n']}</td>"
        f"<td>{_bar_html(d['raw_strict_rate'])} {_match_badge(d['raw_strict_rate'])}</td>"
        f"<td>{_bar_html(d['rkd_strict_rate'])} {_match_badge(d['rkd_strict_rate'])} "
        f"{_delta_html(d['raw_strict_rate'], d['rkd_strict_rate'])}</td>"
        f"<td>{_bar_html(d['rkd_affinity_rate'])} {_match_badge(d['rkd_affinity_rate'])}</td>"
        f"</tr>"
        for cat, d in sorted_cats
    )
    return (
        f"<table>"
        f"<tr><th>Category</th><th>Queries</th><th>Strict (raw)</th>"
        f"<th>Strict (reranked)</th><th>Affinity (reranked)</th></tr>"
        f"{rows_html}"
        f"</table>"
    )


def _fashor_callout_html(fashor_stats: dict) -> str:
    rate = fashor_stats["rkd_strict_rate"]
    per_cat = fashor_stats["per_category"]

    ethnic_cats = {k: v for k, v in per_cat.items() if any(
        kw in k.lower() for kw in ["kurta", "ethnic", "lehenga", "dupatta", "anarkali"]
    )}
    western_cats = {k: v for k, v in per_cat.items() if any(
        kw in k.lower() for kw in ["dress", "top", "jeans", "co-ord", "fashion"]
    )}

    ethnic_rate = (
        float(np.mean([v["rkd_strict_rate"] for v in ethnic_cats.values()]))
        if ethnic_cats else math.nan
    )
    western_rate = (
        float(np.mean([v["rkd_strict_rate"] for v in western_cats.values()]))
        if western_cats else math.nan
    )

    cls = "callout-warn" if rate < 0.5 else ("callout" if rate < 0.8 else "callout-good")
    verdict = "⚠ Low" if rate < 0.5 else ("~ Moderate" if rate < 0.8 else "✓ Strong")

    ethnic_str = _pct(ethnic_rate) if not math.isnan(ethnic_rate) else "n/a"
    western_str = _pct(western_rate) if not math.isnan(western_rate) else "n/a"

    ethnic_rows = "".join(
        f"<li>{_esc(k)}: {_pct(v['rkd_strict_rate'])} ({v['n']} queries)</li>"
        for k, v in sorted(ethnic_cats.items(), key=lambda x: x[1]["rkd_strict_rate"])
    ) or "<li>(no ethnic categories matched keywords)</li>"

    return (
        f'<div class="callout {cls}">'
        f"<h3>Fashor Ethnic Wear Analysis (H&M-trained tower)</h3>"
        f"<p><strong>Reranked strict cat-match:</strong> {_pct(rate)} — {verdict}</p>"
        f"<p style='margin-top:6px'><strong>Ethnic avg (strict):</strong> {ethnic_str}"
        f" &nbsp;|&nbsp; <strong>Western avg (strict):</strong> {western_str}</p>"
        f"<p style='margin-top:8px;font-size:12px;color:#555'>"
        f"The item tower (CLIP ViT-B/32 + SBERT) was trained on H&M western-fashion interactions. "
        f"Kurta/ethnic categories are out-of-distribution for the user tower but CLIP's visual "
        f"embeddings still capture garment shape, color, and texture — so similarity within ethnic "
        f"categories depends entirely on visual+text alignment, not interaction-based fine-tuning."
        f"</p>"
        f"<ul style='margin-top:8px;font-size:12px;padding-left:16px'>{ethnic_rows}</ul>"
        f"</div>"
    )


def build_html(
    all_results: list[QueryResult],
    all_stats: dict[str, dict],
    n_queries: int = 25,
    k: int = 5,
) -> str:
    worst_10 = sorted(all_results, key=lambda r: r.reranked_strict_rate)[:10]
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    toc_html = (
        '<div class="toc">'
        "<strong>Jump to:</strong>"
        "<ul>"
        '<li><a href="#comparison">Before/After Comparison</a></li>'
        '<li><a href="#worst10">Worst 10 Results</a></li>'
        + "".join(f'<li><a href="#{brand}">{brand.title()}</a></li>' for brand in BRANDS)
        + "</ul></div>"
    )

    # Headline stats (reranked)
    all_rkd_strict = [r.reranked_strict_rate for r in all_results]
    all_rkd_affinity = [r.reranked_affinity_rate for r in all_results]
    overall_strict = float(np.mean(all_rkd_strict)) if all_rkd_strict else math.nan
    overall_affinity = float(np.mean(all_rkd_affinity)) if all_rkd_affinity else math.nan
    total_self = sum(1 for r in all_results if r.has_self_match)
    total_dupes = sum(r.n_near_dupes for r in all_results)

    stat_grid_html = (
        '<div class="stat-grid">'
        f'<div class="stat-box"><div class="stat-val">{_pct(overall_strict)}</div>'
        f'<div class="stat-label">Strict cat-match (reranked, {len(all_results)} queries)</div></div>'
        f'<div class="stat-box"><div class="stat-val">{_pct(overall_affinity)}</div>'
        f'<div class="stat-label">Affinity cat-match (reranked, threshold≥{AFFINITY_THRESHOLD})</div></div>'
        f'<div class="stat-box"><div class="stat-val">'
        f'{"⚠ " + str(total_self) if total_self else "✓ 0"}</div>'
        f'<div class="stat-label">Self-match violations</div></div>'
        f'<div class="stat-box"><div class="stat-val">{total_dupes}</div>'
        f'<div class="stat-label">Near-dupe flags (sim≥0.995)</div></div>'
        "</div>"
    )

    brand_sections_html = ""
    for brand in BRANDS:
        brand_results = [r for r in all_results if r.brand == brand]
        if not brand_results:
            continue
        s = all_stats[brand]
        callout = _fashor_callout_html(s) if brand == "fashor" else ""
        cat_table = _per_category_table_html(s["per_category"])
        query_cards = "".join(_query_row_html(r) for r in brand_results)
        brand_sections_html += (
            f'<hr class="section-divider">'
            f'<h2 id="{_esc(brand)}">{_esc(brand.title())} — '
            f'strict {_match_badge(s["rkd_strict_rate"])} | '
            f'affinity {_match_badge(s["rkd_affinity_rate"])} | '
            f'{_inr(s["rkd_mean_price_delta"])} avg |ΔPrice|</h2>'
            f"{callout}"
            f"<h3>Per-Category Breakdown (sorted by reranked strict rate)</h3>"
            f"{cat_table}"
            f"<h3>Query ↦ Top-{k} Neighbors (reranked)</h3>"
            f"{query_cards}"
        )

    worst_html = "".join(_query_row_html(r, is_worst=True) for r in worst_10)

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

  <h2 id="comparison">Before/After Comparison (raw FAISS vs. re-rank)</h2>
  {stat_grid_html}
  {_comparison_table_html(all_stats)}

  <h2 id="worst10">⚠ Worst 10 Results (Lowest Reranked Strict Cat-Match)</h2>
  <p class="meta">10 query items across all brands where /similar (reranked) returned the
     fewest on-category neighbors (strict, exact-label only). Green badge = strict match;
     yellow ~ = affinity-only; red = no match.</p>
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
        description="Before/after similarity-quality inspection for Indian brand catalogs."
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
        s = _brand_stats(results)
        print(
            f"  raw:      strict={_pct(s['raw_strict_rate'])}  "
            f"affinity={_pct(s['raw_affinity_rate'])}  "
            f"dPrice={_inr(s['raw_mean_price_delta'])}"
        )
        print(
            f"  reranked: strict={_pct(s['rkd_strict_rate'])}  "
            f"affinity={_pct(s['rkd_affinity_rate'])}  "
            f"dPrice={_inr(s['rkd_mean_price_delta'])}"
        )
        print()

    all_stats = {
        brand: _brand_stats([r for r in all_results if r.brand == brand])
        for brand in BRANDS
        if any(r.brand == brand for r in all_results)
    }

    guardrails_ok = print_comparison_table(all_stats)

    worst_10 = sorted(all_results, key=lambda r: r.reranked_strict_rate)[:10]
    print("\n=== Worst 10 queries (lowest reranked strict cat-match) ===")
    for i, r in enumerate(worst_10, 1):
        print(f"  {i:2d}. {r.brand:10s} {r.query_category:30s} "
              f"strict(raw={_pct(r.raw_strict_rate)} rkd={_pct(r.reranked_strict_rate)})  "
              f"aid={r.query_article_id}")

    print()
    html_content = build_html(all_results, all_stats, n_queries=args.n_queries, k=args.k)
    args.output.write_text(html_content, encoding="utf-8")
    print(f"Report written to: {args.output}")
    print(f"Open with: start {args.output}")

    if not guardrails_ok:
        print("\n*** GUARDRAIL BREACH — do not commit until resolved ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
