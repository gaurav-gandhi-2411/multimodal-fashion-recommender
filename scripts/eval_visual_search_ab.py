"""scripts/eval_visual_search_ab.py -- A/B eval: current CLIP-512 vs FashionCLIP-512.

Head-to-head comparison of the current raw-CLIP ViT-B/32 visual-search index
(indices/{brand}/visual.faiss) against the FashionCLIP candidate index
(indices/{brand}/visual_fashionclip.faiss) for the SAME stratified sample of
queries per brand, so the two passes are directly comparable.

For each brand (snitch, fashor, powerlook), runs FOUR passes over one
n=100, seed=42 stratified sample:
  1. image query -- current CLIP    (encode_query_image -> visual.faiss)
  2. image query -- FashionCLIP     (FashionCLIPEncoder.encode_batch -> visual_fashionclip.faiss)
  3. text query  -- current CLIP    (encode_query_text -> visual.faiss)
  4. text query  -- FashionCLIP     (FashionCLIPEncoder.encode_text -> visual_fashionclip.faiss)

Reports, per pass:
  - self_retrieval_rate@5   (image-query pass only; source item appears in top-k)
  - category_match_rate@5   (image-query pass; fraction of top-5 whose category
                              equals the query category)
  - category_recall@5       (text-query pass; whether ANY of top-5 matches the
                              query category -- same definition as
                              scripts/eval_style_search.py)
  - per-category breakdown of the above (categories with >= 3 sampled queries)
  - T-Shirt/Shirt confusion (snitch, powerlook): for T-Shirt-category queries,
    what fraction of top-5 slots land in T-Shirt vs Shirt vs other categories
  - Fashor ethnic-wear per-category breakdown (regression check)

This script is READ-ONLY against indices/ and data/ -- it does not build,
modify, or delete any index.

Usage:
    python scripts/eval_visual_search_ab.py
    python scripts/eval_visual_search_ab.py --brands snitch --n-queries 100
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.visual import encode_query_image, encode_query_text  # noqa: E402
from src.encoders.fashion_clip_encoder import FashionCLIPEncoder  # noqa: E402
from src.retrieval.faiss_index import FaissRetriever  # noqa: E402

BRANDS: list[str] = ["snitch", "fashor", "powerlook"]
N_QUERIES_DEFAULT = 100
SEED_DEFAULT = 42
K_IMAGE = 10  # search depth for image-query pass (top-5 reported, +1 buffer for self-retrieval)
K_TEXT = 5  # search depth for text-query pass (category recall@5, mirrors eval_style_search.py)
MIN_CATEGORY_QUERIES = 3  # only report per-category rows backed by >= this many sampled queries

# T-Shirt/Shirt confusion watch-list -- exact category strings verified against
# data/{brand}/items.parquet (snitch uses plurals, powerlook uses singulars).
TSHIRT_SHIRT_CATS: dict[str, tuple[str, str]] = {
    "snitch": ("T-Shirts", "Shirts"),
    "powerlook": ("T-Shirt", "Shirt"),
}

# Fashor ethnic-wear regression watch-list -- exact category strings verified against
# data/fashor/items.parquet. Excludes "Fashion" and "Bottom" (not ethnic-wear categories).
FASHOR_ETHNIC_CATS: list[str] = [
    "Kurtas",
    "Kurta",
    "3P Kurta Set",
    "2P Kurta Set",
    "Kurta Set",
    "Kurti/Tunics",
    "Dresses",
    "Dress",
    "Co-ord Set",
]


# ---------------------------------------------------------------------------
# Shared helpers (adapted from scripts/eval_visual_search.py -- identical logic)
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    """Load config.yaml from repo root (same path the rest of the app uses)."""
    cfg_path = REPO_ROOT / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


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


def _stratified_sample(catalog: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows spread across categories (proportional; min-1 per category).

    Identical logic to scripts/eval_visual_search.py::_stratified_sample so the
    A/B eval uses the same sampling contract as the single-index eval.
    """
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
# Encoder wrappers -- unify current-CLIP and FashionCLIP behind one interface
# ---------------------------------------------------------------------------


def _make_current_image_encoder() -> Callable[[Path], np.ndarray]:
    """Return a Path -> (512,) embedding function using the current CLIP path."""

    def _fn(img_path: Path) -> np.ndarray:
        return encode_query_image(img_path.read_bytes())

    return _fn


def _make_fashionclip_image_encoder(encoder: FashionCLIPEncoder) -> Callable[[Path], np.ndarray]:
    """Return a Path -> (512,) embedding function using FashionCLIP's image tower."""

    def _fn(img_path: Path) -> np.ndarray:
        return encoder.encode_batch([img_path])[0]

    return _fn


def _make_fashionclip_text_encoder(encoder: FashionCLIPEncoder) -> Callable[[str], np.ndarray]:
    """Return a str -> (512,) embedding function using FashionCLIP's text tower."""

    def _fn(text: str) -> np.ndarray:
        return encoder.encode_text([text])[0]

    return _fn


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------


@dataclass
class QueryHit:
    """Result of one query (image or text) against one FAISS index."""

    article_id: int
    category: str
    top5_cats: list[str]  # top-5 result categories, in rank order
    self_retrieved: bool  # source item appears in the searched top-k
    cat_match_fraction: float  # fraction of top-5 whose category == query category
    cat_hit: bool  # whether ANY of top-5 matches the query category (recall-style)


# ---------------------------------------------------------------------------
# Pass runners
# ---------------------------------------------------------------------------


def run_image_pass(
    sample: pd.DataFrame,
    brand: str,
    retriever: FaissRetriever,
    art_map: dict[int, dict],
    encode_fn: Callable[[Path], np.ndarray],
    k: int = K_IMAGE,
) -> list[QueryHit]:
    """Run the image-query pass: encode each sampled item's own image, search, score.

    Mirrors scripts/eval_visual_search.py::eval_brand's per-query loop, generalised
    to accept any Path -> embedding encoder so the same loop drives both the current
    CLIP pass and the FashionCLIP pass.
    """
    results: list[QueryHit] = []
    for _, row in sample.iterrows():
        img_path = _image_path(brand, row)
        if img_path is None:
            continue

        q_aid = int(row["article_id"])
        q_cat = str(row["category"])

        try:
            query_emb = encode_fn(img_path)
        except ValueError:
            continue

        raw = retriever.search(query_emb, k + 1)  # +1 buffer for unambiguous self-retrieval@k
        top_aids_k = [int(aid) for aid, _ in raw][:k]
        top5_cats = [str(art_map.get(aid, {}).get("category", "")) for aid in top_aids_k[:5]]

        cat_match_fraction = (
            sum(1 for c in top5_cats if c == q_cat) / len(top5_cats) if top5_cats else 0.0
        )
        cat_hit = q_cat in top5_cats
        self_retrieved = q_aid in top_aids_k

        results.append(
            QueryHit(
                article_id=q_aid,
                category=q_cat,
                top5_cats=top5_cats,
                self_retrieved=self_retrieved,
                cat_match_fraction=cat_match_fraction,
                cat_hit=cat_hit,
            )
        )
    return results


def run_text_pass(
    sample: pd.DataFrame,
    retriever: FaissRetriever,
    art_map: dict[int, dict],
    encode_fn: Callable[[str], np.ndarray],
    k: int = K_TEXT,
) -> list[QueryHit]:
    """Run the text-query (style-search) pass: encode each item's title, search, score.

    Category recall@5 definition matches scripts/eval_style_search.py: a "hit" is the
    query category appearing ANYWHERE in the top-k results, not a top-5 fraction.
    """
    results: list[QueryHit] = []
    for _, row in sample.iterrows():
        q_aid = int(row["article_id"])
        q_cat = str(row["category"])
        title = str(row.get("title", "")).strip()
        if not title:
            continue

        query_emb = encode_fn(title)
        raw = retriever.search(query_emb, k)
        top_aids = [int(aid) for aid, _ in raw]
        top5_cats = [str(art_map.get(aid, {}).get("category", "")) for aid in top_aids]

        cat_match_fraction = (
            sum(1 for c in top5_cats if c == q_cat) / len(top5_cats) if top5_cats else 0.0
        )
        cat_hit = q_cat in top5_cats
        self_retrieved = q_aid in top_aids

        results.append(
            QueryHit(
                article_id=q_aid,
                category=q_cat,
                top5_cats=top5_cats,
                self_retrieved=self_retrieved,
                cat_match_fraction=cat_match_fraction,
                cat_hit=cat_hit,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def overall_rate(results: list[QueryHit], attr: str) -> float:
    """Mean of a boolean/float QueryHit attribute across all results (0.0 if empty)."""
    if not results:
        return 0.0
    return float(np.mean([getattr(r, attr) for r in results]))


def per_category_rates(
    results: list[QueryHit], attr: str, min_n: int = MIN_CATEGORY_QUERIES
) -> dict[str, tuple[float, int]]:
    """Per-category mean of a QueryHit attribute, restricted to categories with >= min_n queries.

    Returns {category: (mean_rate, n_queries)}.
    """
    if not results:
        return {}
    cats = [r.category for r in results]
    vals = [getattr(r, attr) for r in results]
    df = pd.DataFrame({"category": cats, "val": vals})
    grouped = df.groupby("category")["val"].agg(["mean", "count"])
    return {
        str(cat): (float(row["mean"]), int(row["count"]))
        for cat, row in grouped.iterrows()
        if row["count"] >= min_n
    }


def tshirt_shirt_confusion(
    results: list[QueryHit], tshirt_cat: str, shirt_cat: str
) -> dict[str, float | int]:
    """For T-Shirt-category queries, categorize top-5 slots into T-Shirt/Shirt/other.

    Returns a dict with n_queries, n_slots, and frac_tshirt/frac_shirt/frac_other
    (fractions of all top-5 slots across all T-Shirt queries, not per-query means).
    """
    tshirt_queries = [r for r in results if r.category == tshirt_cat]
    if not tshirt_queries:
        return {
            "n_queries": 0,
            "n_slots": 0,
            "frac_tshirt": 0.0,
            "frac_shirt": 0.0,
            "frac_other": 0.0,
        }

    n_tshirt = n_shirt = n_other = 0
    total_slots = 0
    for r in tshirt_queries:
        for c in r.top5_cats:
            total_slots += 1
            if c == tshirt_cat:
                n_tshirt += 1
            elif c == shirt_cat:
                n_shirt += 1
            else:
                n_other += 1

    return {
        "n_queries": len(tshirt_queries),
        "n_slots": total_slots,
        "frac_tshirt": n_tshirt / total_slots if total_slots else 0.0,
        "frac_shirt": n_shirt / total_slots if total_slots else 0.0,
        "frac_other": n_other / total_slots if total_slots else 0.0,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_overall(label: str, current: list[QueryHit], fclip: list[QueryHit], attr: str) -> None:
    """Print current vs fashionclip overall rate + delta for a QueryHit attribute."""
    c = overall_rate(current, attr) * 100
    f = overall_rate(fclip, attr) * 100
    print(f"    {label:<28} current={c:6.1f}%   fashionclip={f:6.1f}%   delta={f - c:+6.1f}pp")


def print_per_category_table(
    current: list[QueryHit],
    fclip: list[QueryHit],
    attr: str,
    title: str,
    only_cats: list[str] | None = None,
) -> dict[str, tuple[float, float, int]]:
    """Print a per-category current-vs-fashionclip table; returns {cat: (cur, fc, n)}."""
    cur_rates = per_category_rates(current, attr)
    fc_rates = per_category_rates(fclip, attr)
    cats = sorted(set(cur_rates) | set(fc_rates))
    if only_cats is not None:
        cats = [c for c in cats if c in only_cats]

    print(f"    {title}")
    if not cats:
        print(f"      (no categories with >= {MIN_CATEGORY_QUERIES} sampled queries)")
        return {}

    print(f"      {'Category':<18} {'n':>4} {'current':>10} {'fashionclip':>12} {'delta(pp)':>10}")
    result: dict[str, tuple[float, float, int]] = {}
    for cat in cats:
        c_val, c_n = cur_rates.get(cat, (0.0, 0))
        f_val, f_n = fc_rates.get(cat, (0.0, 0))
        n = max(c_n, f_n)
        delta = (f_val - c_val) * 100
        print(f"      {cat:<18} {n:>4} {c_val * 100:>9.1f}% {f_val * 100:>11.1f}% {delta:>+9.1f}")
        result[cat] = (c_val, f_val, n)
    return result


def print_confusion(
    label: str,
    current: list[QueryHit],
    fclip: list[QueryHit],
    tshirt_cat: str,
    shirt_cat: str,
) -> None:
    """Print the T-Shirt/Shirt top-5 slot-composition confusion breakdown."""
    print(f"    {label}")
    for name, results in [("current", current), ("fashionclip", fclip)]:
        d = tshirt_shirt_confusion(results, tshirt_cat, shirt_cat)
        if d["n_queries"] == 0:
            print(f"      {name}: no '{tshirt_cat}' queries in sample")
            continue
        print(
            f"      {name:<12} n_queries={d['n_queries']:<3} "
            f"{tshirt_cat}={d['frac_tshirt'] * 100:5.1f}%  "
            f"{shirt_cat}={d['frac_shirt'] * 100:5.1f}%  "
            f"other={d['frac_other'] * 100:5.1f}%"
        )


# ---------------------------------------------------------------------------
# Per-brand orchestration
# ---------------------------------------------------------------------------


@dataclass
class BrandPasses:
    """The four QueryHit passes for one brand, plus sample metadata."""

    sample: pd.DataFrame
    img_current: list[QueryHit]
    img_fclip: list[QueryHit]
    txt_current: list[QueryHit]
    txt_fclip: list[QueryHit]


def eval_brand(
    brand: str,
    fclip_encoder: FashionCLIPEncoder,
    n_queries: int = N_QUERIES_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> BrandPasses | None:
    """Run all four passes (image/text x current/fashionclip) for one brand."""
    current_dir = REPO_ROOT / "indices" / brand / "visual.faiss"
    fclip_dir = REPO_ROOT / "indices" / brand / "visual_fashionclip.faiss"

    if not current_dir.is_dir():
        print(f"  [SKIP] current index not found: {current_dir}")
        return None
    if not fclip_dir.is_dir():
        print(f"  [SKIP] fashionclip index not found: {fclip_dir}")
        return None

    print("  Loading current CLIP index...", end=" ", flush=True)
    retr_current = FaissRetriever.load(str(current_dir))
    print(f"{retr_current.index.ntotal} vectors, dim={retr_current.index.d}")

    print("  Loading FashionCLIP index...", end=" ", flush=True)
    retr_fclip = FaissRetriever.load(str(fclip_dir))
    print(f"{retr_fclip.index.ntotal} vectors, dim={retr_fclip.index.d}")

    current_ids = {int(a) for a in retr_current.article_ids}
    fclip_ids = {int(a) for a in retr_fclip.article_ids}
    common_ids = current_ids & fclip_ids
    if current_ids != fclip_ids:
        print(
            f"  [WARN] article_id sets differ between indices: "
            f"current={len(current_ids)} fashionclip={len(fclip_ids)} common={len(common_ids)}"
        )

    catalog_path = REPO_ROOT / "data" / brand / "items.parquet"
    catalog = pd.read_parquet(catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)
    art_map: dict[int, dict] = catalog.set_index("article_id").to_dict("index")

    indexed_catalog = catalog[catalog["article_id"].isin(common_ids)]
    sample = _stratified_sample(indexed_catalog, n=n_queries, seed=seed)
    print(
        f"  Sampled {len(sample)} queries across {sample['category'].nunique()} categories "
        f"(items common to both indices: {len(common_ids)})"
    )

    encode_img_current = _make_current_image_encoder()
    encode_img_fclip = _make_fashionclip_image_encoder(fclip_encoder)
    encode_txt_fclip = _make_fashionclip_text_encoder(fclip_encoder)

    print("  Running image-query pass (current CLIP)...", flush=True)
    img_current = run_image_pass(sample, brand, retr_current, art_map, encode_img_current)
    print("  Running image-query pass (FashionCLIP)...", flush=True)
    img_fclip = run_image_pass(sample, brand, retr_fclip, art_map, encode_img_fclip)

    print("  Running text-query pass (current CLIP)...", flush=True)
    txt_current = run_text_pass(sample, retr_current, art_map, encode_query_text)
    print("  Running text-query pass (FashionCLIP)...", flush=True)
    txt_fclip = run_text_pass(sample, retr_fclip, art_map, encode_txt_fclip)

    return BrandPasses(
        sample=sample,
        img_current=img_current,
        img_fclip=img_fclip,
        txt_current=txt_current,
        txt_fclip=txt_fclip,
    )


def print_brand_report(brand: str, passes: BrandPasses) -> None:
    """Print the full per-brand report: overall metrics, breakdowns, confusion, summary."""
    print("\n  [IMAGE QUERY PASS]")
    print_overall("self_retrieval@5", passes.img_current, passes.img_fclip, "self_retrieved")
    print_overall(
        "category_match@5 (overall)", passes.img_current, passes.img_fclip, "cat_match_fraction"
    )
    img_cat_table = print_per_category_table(
        passes.img_current, passes.img_fclip, "cat_match_fraction",
        title="Per-category category_match@5 (image query):",
    )

    print("\n  [TEXT QUERY PASS]")
    print_overall("category_recall@5 (overall)", passes.txt_current, passes.txt_fclip, "cat_hit")
    txt_cat_table = print_per_category_table(
        passes.txt_current, passes.txt_fclip, "cat_hit",
        title="Per-category category_recall@5 (text query):",
    )

    if brand in TSHIRT_SHIRT_CATS:
        tshirt_cat, shirt_cat = TSHIRT_SHIRT_CATS[brand]
        print(f"\n  [T-SHIRT / SHIRT CONFUSION -- {tshirt_cat} vs {shirt_cat}]")
        print_confusion(
            "Image query (top-5 slot composition for T-Shirt-category queries):",
            passes.img_current, passes.img_fclip, tshirt_cat, shirt_cat,
        )
        print_confusion(
            "Text query (top-5 slot composition for T-Shirt-category queries):",
            passes.txt_current, passes.txt_fclip, tshirt_cat, shirt_cat,
        )

    if brand == "fashor":
        print("\n  [FASHOR ETHNIC-WEAR REGRESSION CHECK]")
        print_per_category_table(
            passes.img_current, passes.img_fclip, "cat_match_fraction",
            title="Ethnic-wear category_match@5 (image query):",
            only_cats=FASHOR_ETHNIC_CATS,
        )
        print_per_category_table(
            passes.txt_current, passes.txt_fclip, "cat_hit",
            title="Ethnic-wear category_recall@5 (text query):",
            only_cats=FASHOR_ETHNIC_CATS,
        )

    # --- Consolidated summary table ---
    print("\n  [CONSOLIDATED SUMMARY -- fashionclip minus current, in pp]")
    self_c = overall_rate(passes.img_current, "self_retrieved") * 100
    self_f = overall_rate(passes.img_fclip, "self_retrieved") * 100
    cat_c = overall_rate(passes.img_current, "cat_match_fraction") * 100
    cat_f = overall_rate(passes.img_fclip, "cat_match_fraction") * 100
    txt_c = overall_rate(passes.txt_current, "cat_hit") * 100
    txt_f = overall_rate(passes.txt_fclip, "cat_hit") * 100

    print(f"    {'metric':<38} {'current':>10} {'fashionclip':>12} {'delta(pp)':>10}")

    def _row(label: str, c_val: float, f_val: float) -> None:
        print(f"    {label:<38} {c_val:>9.1f}% {f_val:>11.1f}% {f_val - c_val:>+9.1f}")

    _row("self_retrieval@5 (image)", self_c, self_f)
    _row("category_match@5 overall (image)", cat_c, cat_f)
    for cat, (c_val, f_val, n) in img_cat_table.items():
        _row(f"  category_match@5 [{cat}] (n={n})", c_val * 100, f_val * 100)
    _row("category_recall@5 overall (text)", txt_c, txt_f)
    for cat, (c_val, f_val, n) in txt_cat_table.items():
        _row(f"  category_recall@5 [{cat}] (n={n})", c_val * 100, f_val * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B eval: current CLIP-512 vs FashionCLIP-512 visual/style search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-queries", type=int, default=N_QUERIES_DEFAULT, metavar="N",
                   help="Items sampled per brand (stratified, before image-availability filter)")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed")
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
    print("Visual Search A/B Eval -- current CLIP-512 vs FashionCLIP-512")
    print(f"  brands={args.brands}  n_queries={args.n_queries}  seed={args.seed}")

    cfg = _load_config()
    print("\nLoading FashionCLIP encoder (shared across brands)...", end=" ", flush=True)
    fclip_encoder = FashionCLIPEncoder(cfg)
    print(f"done (device={fclip_encoder.device})")

    for brand in args.brands:
        print(f"\n{'=' * 78}\nBRAND: {brand}\n{'=' * 78}")
        passes = eval_brand(brand, fclip_encoder, n_queries=args.n_queries, seed=args.seed)
        if passes is None:
            continue
        print_brand_report(brand, passes)


if __name__ == "__main__":
    main()
