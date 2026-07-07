"""scripts/build_visual_index_fashionclip.py -- Build per-brand FashionCLIP-512 visual FAISS
indices (A/B candidate against the existing raw-CLIP visual.faiss index).

For each brand: read data/{brand}/items.parquet; for each item with a local image at
data/{brand}/images/{product_id}.jpg, FashionCLIP-encode in batches -> (N, 512), L2-normalise,
build a FAISS IndexFlatIP, and save via FaissRetriever-compatible format to
indices/{brand}/visual_fashionclip.faiss/ (i.e. faiss.index + article_ids.pkl).

This is a parallel candidate index for an A/B test -- it does NOT overwrite or modify
indices/{brand}/visual.faiss (the current production raw-CLIP index built by
scripts/build_visual_index.py).

article_ids in the saved index are the int article_ids of items that HAD images, in
index order. Items lacking a local image are skipped (count reported) -- identical selection
logic to scripts/build_visual_index.py so the two indices cover the same item set.

Usage:
    python scripts/build_visual_index_fashionclip.py
    python scripts/build_visual_index_fashionclip.py --brands snitch,fashor,powerlook
    python scripts/build_visual_index_fashionclip.py --brands snitch --batch-size 32

The script is idempotent: running it again overwrites existing visual_fashionclip.faiss
directories. Seed=42 for reproducibility (encoding is deterministic, but seed is set
defensively).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.encoders.fashion_clip_encoder import FashionCLIPEncoder  # noqa: E402
from src.retrieval.faiss_index import FaissRetriever  # noqa: E402

# Seed for reproducibility (encode_image is deterministic; set defensively).
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


def _image_path(brand: str, row: pd.Series) -> Path | None:
    """Return the local image path for a catalog row, or None if absent.

    Convention (all three brands): data/{brand}/images/{product_id}.jpg.
    Falls back to {article_id}.jpg in case product_id is missing or blank.
    Identical logic to scripts/build_visual_index.py::_image_path so both indices select
    the same item set.
    """
    images_dir = REPO_ROOT / "data" / brand / "images"
    pid = str(row.get("product_id", "")).strip()
    if pid:
        candidate = images_dir / f"{pid}.jpg"
        if candidate.exists():
            return candidate
    # Fallback: article_id (not the primary convention, but defensive).
    aid = str(row.get("article_id", "")).strip()
    if aid:
        candidate = images_dir / f"{aid}.jpg"
        if candidate.exists():
            return candidate
    return None


def build_brand_index(
    brand: str,
    encoder: FashionCLIPEncoder,
    batch_size: int = 64,
) -> None:
    """Build and save the FashionCLIP-512 visual FAISS index for a single brand.

    Args:
        brand: brand slug (e.g. "snitch").
        encoder: pre-initialised FashionCLIPEncoder.
        batch_size: images per forward pass.
    """
    catalog_path = REPO_ROOT / "data" / brand / "items.parquet"
    if not catalog_path.exists():
        print(f"  [SKIP] Catalog not found: {catalog_path}")
        return

    catalog = pd.read_parquet(catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)
    n_items = len(catalog)

    # --- Collect items that have a local image ---
    usable_rows: list[pd.Series] = []
    usable_paths: list[Path] = []
    for _, row in catalog.iterrows():
        img_path = _image_path(brand, row)
        if img_path is not None:
            usable_rows.append(row)
            usable_paths.append(img_path)

    n_usable = len(usable_paths)
    n_skipped = n_items - n_usable
    print(
        f"  {brand}: {n_items} catalog items | {n_usable} with images "
        f"| {n_skipped} skipped (no image)"
    )

    if n_usable == 0:
        print(f"  [SKIP] No images found for {brand}. Cannot build visual index.")
        return

    # --- FashionCLIP-encode in batches ---
    article_ids: list[int] = [int(row["article_id"]) for row in usable_rows]
    all_embs: list[np.ndarray] = []

    for batch_start in range(0, n_usable, batch_size):
        batch_paths = usable_paths[batch_start : batch_start + batch_size]
        batch_embs = encoder.encode_batch(batch_paths)  # (B, 512), already L2-normed
        all_embs.append(batch_embs)
        if (batch_start // batch_size) % 10 == 0:
            pct = min(100, 100 * (batch_start + len(batch_paths)) / n_usable)
            print(f"    encoded {batch_start + len(batch_paths)}/{n_usable} ({pct:.0f}%)",
                  flush=True)

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)  # (N, 512)

    # Defensive re-normalise: encode_batch already normalises, but guard for precision.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embs = embs / norms

    # --- Build and save FaissRetriever-compatible index ---
    output_dir = str(REPO_ROOT / "indices" / brand / "visual_fashionclip.faiss")
    retriever = FaissRetriever(embs, article_ids)
    retriever.save(output_dir)
    print(
        f"  Saved FashionCLIP visual index: {retriever.index.ntotal} vectors "
        f"| dim={retriever.index.d} | path={output_dir}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-brand FashionCLIP-512 visual FAISS indices (A/B candidate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--brands",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=BRANDS_DEFAULT,
        metavar="BRAND1,BRAND2,...",
        help="Comma-separated list of brand slugs to process.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Images per FashionCLIP forward pass.",
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

    print(f"\nBuilding FashionCLIP visual indices for brands: {args.brands}")
    start = time.perf_counter()
    for brand in args.brands:
        print(f"\n[{brand}]")
        build_brand_index(brand, encoder, batch_size=args.batch_size)
    elapsed = time.perf_counter() - start

    print(f"\nDone. Total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
