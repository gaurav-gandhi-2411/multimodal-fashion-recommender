"""scripts/eval_attributes.py -- Automatic, full-catalog cross-validation eval for the
zero-shot FashionCLIP attribute extraction (app/attributes.py, data/{brand}/attributes.json).

This is deliberately NOT a model-quality confirmation: it cross-checks the visual
classifier's predictions against two INDEPENDENT text-derived signals already present in
each catalogue, so we can see plainly which of the 4 attributes (color, pattern, fabric,
occasion) are trustworthy and which are not.

Two signal sources, at full catalogue scale (read-only against attributes.json):

1. Text-keyword cross-validation (color, pattern, fabric): for each item, check whether
   EXACTLY ONE taxonomy label for a category appears as a whole-word/phrase match in
   title + description. That defines an unambiguous, text-derivable ground truth for that
   item/category. Coverage = how much of the catalogue has such a label at all; accuracy =
   agreement between the model's prediction and that label, within the unambiguous subset.

2. Occasion cross-validation against app/occasion.py::tag_occasions -- an already-in-
   production, hand-built lexicon tagger. For items where the tagger returns a non-empty
   set, check whether the model's predicted occasion is a MEMBER of that set (an item can
   plausibly belong to multiple occasion buckets via text, so membership -- not exact
   single-label match -- is the fair comparison). Also computes each brand's MAJORITY-CLASS
   baseline (always predict the model's single most-frequent predicted occasion) on the
   SAME evaluable subset, so a classifier that merely matches "always guess casual" is
   called out explicitly rather than looking like a real win.

3. Confidence sanity check: mean {category}_confidence for correct vs incorrect predictions,
   within each evaluable subset -- flags whether the confidence score is a usable trust
   filter at all.

4. Spot-check sample prep: writes a stratified (seed=42), image-path-verified sample of
   18 items per brand (54 total) to reports/attribute_spotcheck_sample.json for a SEPARATE
   manual visual review (this script does not judge images itself).

This script is read-only against app/attributes.py, scripts/extract_attributes.py, and the
already-built data/{brand}/attributes.json files -- it does not modify or rebuild any of them.

Usage:
    python scripts/eval_attributes.py
    python scripts/eval_attributes.py --brands snitch,fashor
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.attributes import ATTRIBUTE_TAXONOMY, AttributeIndex, load_attribute_index  # noqa: E402
from app.occasion import DEFAULT_OCCASION_LEXICON, tag_occasions  # noqa: E402

BRANDS: list[str] = ["snitch", "fashor", "powerlook"]
TEXT_CATEGORIES: list[str] = ["color", "pattern", "fabric"]  # occasion handled separately
SPOTCHECK_N_PER_BRAND: int = 18
SPOTCHECK_SEED: int = 42

# ---------------------------------------------------------------------------
# Text-keyword matching
# ---------------------------------------------------------------------------

_LABEL_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _label_pattern(label: str) -> re.Pattern[str]:
    """Compile (and cache) a whole-word/phrase, case-insensitive regex for *label*.

    ``re.escape`` handles multi-word labels like "polka dot" / "animal print" correctly:
    the escaped literal space keeps the phrase intact, and the surrounding ``\\b`` anchors
    still apply to the phrase's outer edges.
    """
    if label not in _LABEL_PATTERN_CACHE:
        _LABEL_PATTERN_CACHE[label] = re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
    return _LABEL_PATTERN_CACHE[label]


def find_text_labels(text: str, labels: list[str]) -> list[str]:
    """Return every taxonomy label from *labels* found as a whole-word/phrase match in *text*."""
    return [label for label in labels if _label_pattern(label).search(text)]


def unambiguous_text_label(text: str, labels: list[str]) -> str | None:
    """Return the single taxonomy label found in *text*, or None if zero or more than one
    label matched (ambiguous items are excluded from the evaluable subset by design)."""
    found = find_text_labels(text, labels)
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# Category (color/pattern/fabric) text cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CategoryEvalResult:
    """Text-keyword cross-validation result for one attribute category (one brand, or pooled)."""

    category: str
    n_total: int  # full catalogue size (coverage denominator)
    n_unambiguous: int  # items with exactly one text-derivable label (coverage numerator)
    n_evaluated: int  # unambiguous AND has a model prediction (accuracy denominator)
    n_correct: int  # predicted label == text-derived label, within n_evaluated
    correct_confidences: list[float] = field(default_factory=list)
    incorrect_confidences: list[float] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the full catalogue with an unambiguous text-derivable label."""
        return self.n_unambiguous / self.n_total if self.n_total else 0.0

    @property
    def accuracy(self) -> float:
        """Agreement rate between predicted and text-derived label, within the evaluable subset."""
        return self.n_correct / self.n_evaluated if self.n_evaluated else 0.0


def eval_category_text_xval(
    items: pd.DataFrame, attr_index: AttributeIndex, category: str
) -> CategoryEvalResult:
    """Cross-validate one attribute category's predictions against unambiguous text labels.

    Args:
        items: brand catalogue with at least ``article_id``, ``title``, ``description``.
        attr_index: brand's loaded attributes.json (AttributeIndex).
        category: one of "color", "pattern", "fabric" (ATTRIBUTE_TAXONOMY key).

    Returns:
        CategoryEvalResult with raw counts and confidence lists (not just rates), so callers
        can pool results across brands by summing counts rather than averaging rates.
    """
    labels = ATTRIBUTE_TAXONOMY[category]
    n_total = 0
    n_unambiguous = 0
    n_evaluated = 0
    n_correct = 0
    correct_conf: list[float] = []
    incorrect_conf: list[float] = []

    for row in items.itertuples(index=False):
        n_total += 1
        text = f"{row.title} {row.description}"
        gt = unambiguous_text_label(text, labels)
        if gt is None:
            continue
        n_unambiguous += 1

        entry = attr_index.get(str(row.article_id))
        if entry is None:
            continue  # item skipped at extraction time (e.g. missing image) -- no prediction
        n_evaluated += 1

        pred = entry[category]
        conf = float(entry[f"{category}_confidence"])
        if pred == gt:
            n_correct += 1
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)

    return CategoryEvalResult(
        category=category,
        n_total=n_total,
        n_unambiguous=n_unambiguous,
        n_evaluated=n_evaluated,
        n_correct=n_correct,
        correct_confidences=correct_conf,
        incorrect_confidences=incorrect_conf,
    )


def pool_category_results(results: list[CategoryEvalResult], category: str) -> CategoryEvalResult:
    """Pool per-brand CategoryEvalResults into one (sums raw counts -- never averages rates)."""
    return CategoryEvalResult(
        category=category,
        n_total=sum(r.n_total for r in results),
        n_unambiguous=sum(r.n_unambiguous for r in results),
        n_evaluated=sum(r.n_evaluated for r in results),
        n_correct=sum(r.n_correct for r in results),
        correct_confidences=[c for r in results for c in r.correct_confidences],
        incorrect_confidences=[c for r in results for c in r.incorrect_confidences],
    )


# ---------------------------------------------------------------------------
# Occasion cross-validation (against app/occasion.py's trusted lexicon tagger)
# ---------------------------------------------------------------------------


@dataclass
class OccasionEvalResult:
    """Occasion cross-validation result (one brand, or pooled) against tag_occasions()."""

    n_total: int
    n_covered: int  # tag_occasions() found a non-empty set (coverage numerator)
    n_evaluated: int  # covered AND has a model prediction (accuracy denominator)
    n_correct: int  # predicted occasion is a MEMBER of the tagger's set
    correct_confidences: list[float] = field(default_factory=list)
    incorrect_confidences: list[float] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_covered / self.n_total if self.n_total else 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_evaluated if self.n_evaluated else 0.0


def eval_occasion_text_xval(
    items: pd.DataFrame,
    attr_index: AttributeIndex,
    lexicon: dict[str, list[str]] | None = None,
) -> OccasionEvalResult:
    """Cross-validate occasion predictions against app/occasion.py::tag_occasions() membership.

    Args:
        items: brand catalogue with at least ``article_id``, ``title``, ``description``.
        attr_index: brand's loaded attributes.json (AttributeIndex).
        lexicon: occasion keyword lexicon; defaults to DEFAULT_OCCASION_LEXICON.

    Returns:
        OccasionEvalResult with raw counts and confidence lists.
    """
    n_total = 0
    n_covered = 0
    n_evaluated = 0
    n_correct = 0
    correct_conf: list[float] = []
    incorrect_conf: list[float] = []

    for row in items.itertuples(index=False):
        n_total += 1
        tags = tag_occasions(str(row.title), str(row.description), lexicon=lexicon)
        if not tags:
            continue
        n_covered += 1

        entry = attr_index.get(str(row.article_id))
        if entry is None:
            continue
        n_evaluated += 1

        pred = entry["occasion"]
        conf = float(entry["occasion_confidence"])
        if pred in tags:
            n_correct += 1
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)

    return OccasionEvalResult(
        n_total=n_total,
        n_covered=n_covered,
        n_evaluated=n_evaluated,
        n_correct=n_correct,
        correct_confidences=correct_conf,
        incorrect_confidences=incorrect_conf,
    )


def pool_occasion_results(results: list[OccasionEvalResult]) -> OccasionEvalResult:
    """Pool per-brand OccasionEvalResults into one (sums raw counts)."""
    return OccasionEvalResult(
        n_total=sum(r.n_total for r in results),
        n_covered=sum(r.n_covered for r in results),
        n_evaluated=sum(r.n_evaluated for r in results),
        n_correct=sum(r.n_correct for r in results),
        correct_confidences=[c for r in results for c in r.correct_confidences],
        incorrect_confidences=[c for r in results for c in r.incorrect_confidences],
    )


def majority_predicted_label(attr_index: AttributeIndex, category: str) -> tuple[str, float]:
    """Return (label, fraction_of_catalog) for the model's single most-frequent PREDICTED
    label in *category*, per "the extraction run's own label distribution"."""
    counts = Counter(entry[category] for entry in attr_index.values())
    if not counts:
        return "", 0.0
    label, n = counts.most_common(1)[0]
    return label, n / len(attr_index)


def majority_baseline_accuracy(
    items: pd.DataFrame,
    attr_index: AttributeIndex,
    lexicon: dict[str, list[str]] | None,
    majority_label: str,
) -> tuple[int, int]:
    """Accuracy of a dumb baseline that ALWAYS predicts *majority_label*, evaluated on the
    EXACT same denominator as the real occasion accuracy: items where tag_occasions() found
    non-empty signal AND the model has a prediction for that item (n_evaluated). This keeps
    the comparison apples-to-apples -- both numbers are computed over the identical subset.

    Returns:
        (n_correct, n_evaluated).
    """
    n_evaluated = 0
    n_correct = 0
    for row in items.itertuples(index=False):
        tags = tag_occasions(str(row.title), str(row.description), lexicon=lexicon)
        if not tags:
            continue
        if attr_index.get(str(row.article_id)) is None:
            continue
        n_evaluated += 1
        if majority_label in tags:
            n_correct += 1
    return n_correct, n_evaluated


# ---------------------------------------------------------------------------
# Confidence sanity check
# ---------------------------------------------------------------------------


def confidence_gap(
    correct: list[float], incorrect: list[float]
) -> tuple[float | None, float | None]:
    """Return (mean_correct_confidence, mean_incorrect_confidence); None where a list is empty."""
    mean_correct = float(np.mean(correct)) if correct else None
    mean_incorrect = float(np.mean(incorrect)) if incorrect else None
    return mean_correct, mean_incorrect


# ---------------------------------------------------------------------------
# Spot-check sample prep (stratified, seed=42, image-verified)
# ---------------------------------------------------------------------------


def _resolve_image_path(images_dir: Path, product_id: str, article_id: str) -> Path | None:
    """Local image path for a catalogue row, or None if absent.

    Convention (all three brands, matches app/ingestion/pipeline.py and
    scripts/eval_visual_search.py): data/{brand}/images/{product_id}.jpg, falling back to
    {article_id}.jpg.
    """
    candidate = images_dir / f"{product_id}.jpg"
    if candidate.exists():
        return candidate
    candidate = images_dir / f"{article_id}.jpg"
    if candidate.exists():
        return candidate
    return None


def _stratified_category_sample(
    catalog: pd.DataFrame, n: int, seed: int
) -> pd.DataFrame:
    """Sample n rows spread proportionally across catalog['category'] (min-1 per category)."""
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

    # Top up any shortfall left by proportional-rounding-down (e.g. n=18 across 11 categories
    # can round to 17): add one slot at a time to categories with spare rows, largest first.
    for cat in cat_counts.index.tolist():
        if total >= n:
            break
        max_extra = len(catalog[catalog["category"] == cat]) - slots[cat]
        if max_extra > 0:
            slots[cat] += 1
            total += 1

    samples = []
    for cat, count in slots.items():
        rows = catalog[catalog["category"] == cat]
        n_sample = min(count, len(rows))
        samples.append(rows.sample(n_sample, random_state=int(rng.integers(1_000_000))))

    return pd.concat(samples).head(n).reset_index(drop=True)


def build_spotcheck_sample(
    brand: str,
    items: pd.DataFrame,
    attr_index: AttributeIndex,
    images_dir: Path,
    n: int = SPOTCHECK_N_PER_BRAND,
    seed: int = SPOTCHECK_SEED,
) -> list[dict]:
    """Build a self-contained, stratified, image-verified spot-check sample for one brand.

    Only items that (a) have a resolvable local image file and (b) have model predictions in
    *attr_index* are eligible -- this pre-filter is how "skip and resample if the image is
    missing" is satisfied: sampling only ever draws from an already-verified-eligible pool,
    so every returned entry's image_path is guaranteed to exist.

    Args:
        brand: brand slug.
        items: brand catalogue (article_id, product_id, title, description, category, ...).
        attr_index: brand's loaded attributes.json.
        images_dir: data/{brand}/images directory.
        n: number of items to sample.
        seed: RNG seed for reproducibility.

    Returns:
        List of up to n dicts: brand, article_id, product_id, image_path, title, attributes.
    """
    eligible_rows = []
    for row in items.itertuples(index=False):
        aid = str(row.article_id)
        entry = attr_index.get(aid)
        if entry is None:
            continue
        img_path = _resolve_image_path(images_dir, str(row.product_id), aid)
        if img_path is None:
            continue
        eligible_rows.append(row)

    if not eligible_rows:
        return []

    eligible_df = pd.DataFrame(eligible_rows)
    sample_n = min(n, len(eligible_df))
    sampled = _stratified_category_sample(eligible_df, sample_n, seed)

    out: list[dict] = []
    for row in sampled.itertuples(index=False):
        aid = str(row.article_id)
        entry = attr_index[aid]
        img_path = _resolve_image_path(images_dir, str(row.product_id), aid)
        assert img_path is not None and img_path.exists()  # guaranteed by eligibility filter above
        out.append(
            {
                "brand": brand,
                "article_id": row.article_id,
                "product_id": row.product_id,
                "image_path": str(img_path),
                "title": row.title,
                "attributes": {
                    category: {
                        "label": entry[category],
                        "confidence": entry[f"{category}_confidence"],
                    }
                    for category in ATTRIBUTE_TAXONOMY
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_conf(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "n/a"


def print_category_report(
    brand_results: dict[str, CategoryEvalResult], pooled: CategoryEvalResult
) -> None:
    """Print coverage/accuracy/confidence-gap for one text-cross-validated category."""
    category = pooled.category
    print(f"\n=== {category.upper()} (text-keyword cross-validation) ===")
    for brand, r in brand_results.items():
        mean_c, mean_i = confidence_gap(r.correct_confidences, r.incorrect_confidences)
        print(
            f"  {brand:12s} coverage={_fmt_pct(r.coverage):>7s} ({r.n_unambiguous}/{r.n_total})  "
            f"accuracy={_fmt_pct(r.accuracy):>7s} ({r.n_correct}/{r.n_evaluated})  "
            f"conf[correct]={_fmt_conf(mean_c)}  conf[incorrect]={_fmt_conf(mean_i)}"
        )
    mean_c, mean_i = confidence_gap(pooled.correct_confidences, pooled.incorrect_confidences)
    print(
        f"  {'POOLED':12s} coverage={_fmt_pct(pooled.coverage):>7s} "
        f"({pooled.n_unambiguous}/{pooled.n_total})  "
        f"accuracy={_fmt_pct(pooled.accuracy):>7s} ({pooled.n_correct}/{pooled.n_evaluated})  "
        f"conf[correct]={_fmt_conf(mean_c)}  conf[incorrect]={_fmt_conf(mean_i)}"
    )


def print_occasion_report(
    brand_results: dict[str, OccasionEvalResult],
    pooled: OccasionEvalResult,
    brand_majority: dict[str, tuple[str, float]],
    brand_baseline: dict[str, tuple[int, int]],
) -> None:
    """Print occasion coverage/accuracy plus the majority-class-baseline honesty check."""
    print("\n=== OCCASION (cross-validation against app/occasion.py::tag_occasions) ===")
    for brand, r in brand_results.items():
        mean_c, mean_i = confidence_gap(r.correct_confidences, r.incorrect_confidences)
        maj_label, maj_frac = brand_majority[brand]
        base_correct, base_evaluated = brand_baseline[brand]
        base_acc = base_correct / base_evaluated if base_evaluated else 0.0
        print(
            f"  {brand:12s} coverage={_fmt_pct(r.coverage):>7s} ({r.n_covered}/{r.n_total})  "
            f"accuracy={_fmt_pct(r.accuracy):>7s} ({r.n_correct}/{r.n_evaluated})  "
            f"conf[correct]={_fmt_conf(mean_c)}  conf[incorrect]={_fmt_conf(mean_i)}"
        )
        print(
            f"               majority-predicted-class='{maj_label}' "
            f"({_fmt_pct(maj_frac)} of {brand}'s own predictions)  "
            f"'always predict {maj_label}' baseline accuracy on the SAME evaluable subset = "
            f"{_fmt_pct(base_acc)} ({base_correct}/{base_evaluated})"
        )
        gap = r.accuracy - base_acc
        if gap < 0.05:
            print(
                f"               ** HONESTY FLAG: visual classifier accuracy is only "
                f"{gap * 100:+.1f}pp over the majority-class baseline for {brand} -- "
                "not meaningfully better than always guessing the majority class."
            )
    mean_c, mean_i = confidence_gap(pooled.correct_confidences, pooled.incorrect_confidences)
    print(
        f"  {'POOLED':12s} coverage={_fmt_pct(pooled.coverage):>7s} "
        f"({pooled.n_covered}/{pooled.n_total})  "
        f"accuracy={_fmt_pct(pooled.accuracy):>7s} ({pooled.n_correct}/{pooled.n_evaluated})  "
        f"conf[correct]={_fmt_conf(mean_c)}  conf[incorrect]={_fmt_conf(mean_i)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automatic, full-catalogue cross-validation eval for zero-shot attributes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--brands",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=BRANDS,
        metavar="BRAND1,BRAND2,...",
        help="Comma-separated list of brand slugs to process.",
    )
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    brand_items: dict[str, pd.DataFrame] = {}
    brand_attrs: dict[str, AttributeIndex] = {}
    for brand in args.brands:
        items_path = REPO_ROOT / "data" / brand / "items.parquet"
        attrs_path = REPO_ROOT / "data" / brand / "attributes.json"
        if not items_path.exists():
            print(f"[SKIP] {brand}: {items_path} not found")
            continue
        brand_items[brand] = pd.read_parquet(items_path)
        brand_attrs[brand] = load_attribute_index(attrs_path)
        print(
            f"[{brand}] catalog n={len(brand_items[brand])}  "
            f"attributes.json n={len(brand_attrs[brand])}"
        )

    # --- 1. Text-keyword cross-validation: color, pattern, fabric ---
    for category in TEXT_CATEGORIES:
        per_brand: dict[str, CategoryEvalResult] = {
            brand: eval_category_text_xval(brand_items[brand], brand_attrs[brand], category)
            for brand in brand_items
        }
        pooled = pool_category_results(list(per_brand.values()), category)
        print_category_report(per_brand, pooled)

    # --- 2. Occasion cross-validation + majority-class baseline honesty check ---
    per_brand_occ: dict[str, OccasionEvalResult] = {
        brand: eval_occasion_text_xval(
            brand_items[brand], brand_attrs[brand], DEFAULT_OCCASION_LEXICON
        )
        for brand in brand_items
    }
    pooled_occ = pool_occasion_results(list(per_brand_occ.values()))
    brand_majority = {
        brand: majority_predicted_label(brand_attrs[brand], "occasion") for brand in brand_items
    }
    brand_baseline = {
        brand: majority_baseline_accuracy(
            brand_items[brand],
            brand_attrs[brand],
            DEFAULT_OCCASION_LEXICON,
            brand_majority[brand][0],
        )
        for brand in brand_items
    }
    print_occasion_report(per_brand_occ, pooled_occ, brand_majority, brand_baseline)

    # --- 4. Spot-check sample prep ---
    print("\n=== SPOT-CHECK SAMPLE PREP ===")
    all_samples: list[dict] = []
    for brand in brand_items:
        images_dir = REPO_ROOT / "data" / brand / "images"
        samples = build_spotcheck_sample(brand, brand_items[brand], brand_attrs[brand], images_dir)
        verified = [s for s in samples if Path(s["image_path"]).exists()]
        print(
            f"  {brand}: {len(verified)}/{SPOTCHECK_N_PER_BRAND} sampled "
            "(all image paths verified)"
        )
        all_samples.extend(verified)

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "attribute_spotcheck_sample.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_samples, fh, indent=2, default=str)
    print(f"  Wrote {len(all_samples)} entries to {out_path}")


if __name__ == "__main__":
    main()
