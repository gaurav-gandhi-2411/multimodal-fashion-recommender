"""Apply app/ingestion/category_normalize.py's per-brand canonicalization map to an
already-ingested brand's catalog parquet, in place.

This only rewrites the `category` column of `data/{brand}/items.parquet` -- it does not
touch embeddings, the FAISS index, or any other column. Category is metadata used by
rerank category-affinity matching and category-match evals; it is never fed into the CLIP/
SBERT encoders, so no re-ingestion or re-encoding is needed.

Usage:
    python scripts/apply_category_normalization.py --brand fashor
    python scripts/apply_category_normalization.py --brand virgio
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app.ingestion.category_normalize import CATEGORY_CANONICALIZATION, canonicalize_category


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", required=True, help="Brand slug (must have a registered map).")
    parser.add_argument(
        "--output-base", default=".", help="Base directory (default: current directory)."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    brand = args.brand

    if brand not in CATEGORY_CANONICALIZATION:
        print(
            f"ERROR: no category canonicalization registered for brand {brand!r}.",
            file=sys.stderr,
        )
        return 1

    catalog_path = Path(args.output_base) / "data" / brand / "items.parquet"
    if not catalog_path.exists():
        print(f"ERROR: catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(catalog_path)
    before_counts = df["category"].value_counts().to_dict()

    df["category"] = df["category"].apply(lambda c: canonicalize_category(brand, c))
    after_counts = df["category"].value_counts().to_dict()

    changed = {k: v for k, v in CATEGORY_CANONICALIZATION[brand].items() if k in before_counts}
    if not changed:
        print(f"No matching raw labels found in {brand}'s catalog -- nothing to change.")
        return 0

    print(f"Category normalization for brand {brand!r}:")
    for raw, canonical in changed.items():
        n_raw = before_counts.get(raw, 0)
        n_canonical_before = before_counts.get(canonical, 0)
        n_canonical_after = after_counts.get(canonical, 0)
        print(
            f"  {raw!r} ({n_raw} items) -> {canonical!r}: "
            f"{n_canonical_before} -> {n_canonical_after} items"
        )

    df.to_parquet(catalog_path, index=False)
    print(f"Wrote {len(df)} rows to {catalog_path} (category column only changed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
