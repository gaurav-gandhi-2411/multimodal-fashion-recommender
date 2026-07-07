"""scripts/extract_attributes.py -- Batch zero-shot catalog attribute extraction.

For each brand: load the existing FashionCLIP visual index (indices/{brand}/
visual_fashionclip.faiss), reconstruct the full image-embedding matrix (no re-encoding),
classify every item against the 4-category attribute taxonomy (app/attributes.py), and
save the result to data/{brand}/attributes.json -- the same JSON-keyed-by-item-id-string
convention as data/{brand}/colors.json.

This does NOT touch the two-tower model, /recommend, /similar, /complete, or
src/models/ -- it is a pure additive read of the already-built FashionCLIP visual index.

Usage:
    python scripts/extract_attributes.py
    python scripts/extract_attributes.py --brands snitch,fashor,powerlook
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.attributes import (  # noqa: E402
    ATTRIBUTE_RELIABILITY,
    ATTRIBUTE_TAXONOMY,
    AttributeIndex,
    build_attribute_index,
    classify_embeddings,
)
from src.encoders.fashion_clip_encoder import FashionCLIPEncoder  # noqa: E402
from src.retrieval.faiss_index import FaissRetriever  # noqa: E402

# Seed for reproducibility (classification is deterministic; set defensively).
_SEED = 42

BRANDS_DEFAULT: list[str] = ["snitch", "fashor", "powerlook"]


def _set_seed() -> None:
    """Fix Python / NumPy / Torch seeds for deterministic behaviour."""
    random.seed(_SEED)
    np.random.seed(_SEED)  # noqa: NPY002
    torch.manual_seed(_SEED)


def _load_config() -> dict:
    """Load config.yaml from repo root."""
    cfg_path = REPO_ROOT / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def _print_distributions(brand: str, index: AttributeIndex) -> None:
    """Print a per-brand, per-category label distribution for a human sanity check."""
    counters: dict[str, Counter] = {cat: Counter() for cat in ATTRIBUTE_TAXONOMY}
    for entry in index.values():
        for cat in ATTRIBUTE_TAXONOMY:
            counters[cat][entry[cat]] += 1

    print(f"  [{brand}] label distributions (n={len(index)}):")
    for cat, counter in counters.items():
        dist_str = ", ".join(f"{label}={count}" for label, count in counter.most_common())
        reliability = ATTRIBUTE_RELIABILITY.get(cat, "unknown")
        print(f"    {cat} [reliability={reliability}]: {dist_str}")

    # Honest reminder every time this script runs -- see app/attributes.py::
    # ATTRIBUTE_RELIABILITY for the cited eval evidence behind these tiers.
    print(
        "  Reliability verdict: color=validated (best of 4); "
        "pattern/fabric/occasion=experimental (fabric barely above random; occasion "
        "worse than a majority-class baseline for 2/3 brands). Do not present "
        "experimental tags as reliable."
    )


def extract_brand_attributes(brand: str, encoder: FashionCLIPEncoder) -> None:
    """Classify every item in a brand's FashionCLIP visual index and save attributes.json.

    Args:
        brand: brand slug (e.g. "snitch").
        encoder: pre-initialised FashionCLIPEncoder (reused across brands).
    """
    index_dir = REPO_ROOT / "indices" / brand / "visual_fashionclip.faiss"
    if not index_dir.is_dir():
        print(f"  [SKIP] FashionCLIP visual index not found: {index_dir}")
        return

    retriever = FaissRetriever.load(str(index_dir))
    n_total = retriever.index.ntotal
    if n_total == 0:
        print(f"  [SKIP] {brand}: visual index is empty")
        return

    image_embeddings: np.ndarray = retriever.index.reconstruct_n(0, n_total).astype(np.float32)
    article_ids: list[int] = [int(aid) for aid in retriever.article_ids]

    print(f"  {brand}: classifying {n_total} items across {len(ATTRIBUTE_TAXONOMY)} categories")
    classified = classify_embeddings(image_embeddings, encoder)
    index = build_attribute_index(article_ids, classified)

    output_path = REPO_ROOT / "data" / brand / "attributes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w") as fh:
        json.dump(index, fh)
    print(f"  Saved {len(index)} item attribute tags to {output_path}")

    _print_distributions(brand, index)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch zero-shot catalog attribute extraction via FashionCLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--brands",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=BRANDS_DEFAULT,
        metavar="BRAND1,BRAND2,...",
        help="Comma-separated list of brand slugs to process.",
    )
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    _set_seed()

    cfg = _load_config()
    device_str = cfg["encoders"].get("device", "cuda")
    print(f"Device: {device_str}")

    print("Loading FashionCLIP (patrickjohncyh/fashion-clip) encoder...")
    encoder = FashionCLIPEncoder(cfg)
    print(f"Encoder ready on {encoder.device}")

    print(f"\nExtracting attributes for brands: {args.brands}")
    start = time.perf_counter()
    for brand in args.brands:
        print(f"\n[{brand}]")
        extract_brand_attributes(brand, encoder)
    elapsed = time.perf_counter() - start

    print(f"\nDone. Total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
