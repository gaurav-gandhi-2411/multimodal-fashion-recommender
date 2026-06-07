from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Enables `from src.* import ...` inside helper function bodies below.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ingestion.images import download_images
from app.ingestion.schema import CatalogRow

logger = logging.getLogger(__name__)


# ── lazy-import helpers (patchable by name, never load at collection time) ────


def _build_img_encoder(cfg: dict):
    from src.encoders.image_encoder import ImageEncoder  # noqa: PLC0415

    return ImageEncoder(cfg)


def _build_txt_encoder(cfg: dict):
    from src.encoders.text_encoder import TextEncoder  # noqa: PLC0415

    return TextEncoder(cfg)


def _load_two_tower(checkpoint_path: Path, device: torch.device):
    from src.models.two_tower import TwoTowerModel  # noqa: PLC0415

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _fuse_embeddings(
    model,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    from src.training.train import encode_all_items  # noqa: PLC0415

    return encode_all_items(model, img_emb, txt_emb, device, batch_size=batch_size)


def _build_retriever(fused_emb: np.ndarray, article_ids: list[str]):
    from src.retrieval.faiss_index import FaissRetriever  # noqa: PLC0415

    return FaissRetriever(fused_emb, article_ids, metric="cosine")


def _item_text(row: CatalogRow) -> str:
    return f"{row.title}. {row.description}. Category: {row.category}."


def run_catalog_pipeline(
    items: list[CatalogRow],
    brand: str,
    *,
    output_base: Path = Path("."),
    checkpoint_path: Path = Path("checkpoints/best.pt"),
    config_path: Path = Path("config.yaml"),
    max_download_workers: int = 8,
    image_batch_size: int = 64,
    text_batch_size: int = 128,
    item_encode_batch_size: int = 512,
) -> Path:
    """
    Full catalog ingestion pipeline: fetch → encode → index → write brand YAML.

    Downloads images (resumable; writes failed_images.json for misses), encodes
    via CLIP ViT-B/32 + SBERT all-MiniLM-L6-v2, fuses through the trained
    ItemTower to 256-dim unit vectors, builds a FAISS IndexFlatIP, and writes
    brands/<brand>.yaml so the Phase 1 API serves the brand immediately with no
    interaction data (cold-start baseline).

    article_ids are sequential 1-indexed strings in FAISS and ints in the parquet
    so the existing registry code (int(aid)) round-trips correctly.

    Returns the path to the written brands/<brand>.yaml.
    """
    if not items:
        raise ValueError("items list is empty — nothing to ingest")

    # ── output paths ──────────────────────────────────────────────────────────
    data_dir = output_base / "data" / brand
    images_dir = data_dir / "images"
    catalog_parquet_path = data_dir / "items.parquet"
    index_dir = output_base / "indices" / brand / "active.faiss"
    embeddings_path = output_base / "indices" / brand / "item_emb.npy"
    brands_dir = output_base / "brands"
    brand_yaml_path = brands_dir / f"{brand}.yaml"

    for d in (data_dir, images_dir, index_dir.parent, brands_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: download images ────────────────────────────────────────────────
    logger.info("Step 1/6 — downloading images for %d items", len(items))
    download_results = download_images(items, images_dir, max_workers=max_download_workers)
    n_failed = sum(1 for v in download_results.values() if v is None)
    if n_failed:
        logger.warning(
            "%d/%d images failed to download (see %s)",
            n_failed,
            len(items),
            images_dir / "failed_images.json",
        )

    image_paths: list[Path] = [
        download_results.get(item.product_id) or (images_dir / f"{item.product_id}.jpg")
        for item in items
    ]

    # ── Step 2: load config ───────────────────────────────────────────────────
    with open(config_path) as fh:
        cfg: dict = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["encoders"]["device"] = str(device)

    # ── Step 3: CLIP image encoding ───────────────────────────────────────────
    logger.info("Step 2/6 — CLIP image encoding (device=%s)", device)
    img_encoder = _build_img_encoder(cfg)
    img_batches = []
    for start in range(0, len(image_paths), image_batch_size):
        img_batches.append(img_encoder.encode_batch(image_paths[start : start + image_batch_size]))
    img_emb: np.ndarray = np.vstack(img_batches).astype(np.float32)
    logger.info("Image embeddings: %s", img_emb.shape)

    # ── Step 4: SBERT text encoding ───────────────────────────────────────────
    logger.info("Step 3/6 — SBERT text encoding")
    txt_encoder = _build_txt_encoder(cfg)
    texts = [_item_text(item) for item in items]
    txt_batches = []
    for start in range(0, len(texts), text_batch_size):
        txt_batches.append(txt_encoder.encode_batch(texts[start : start + text_batch_size]))
    txt_emb: np.ndarray = np.vstack(txt_batches).astype(np.float32)
    logger.info("Text embeddings: %s", txt_emb.shape)

    # ── Step 5: ItemTower fusion → 256-dim ────────────────────────────────────
    logger.info("Step 4/6 — ItemTower fusion (checkpoint: %s)", checkpoint_path)
    model = _load_two_tower(checkpoint_path, device)
    fused_emb: np.ndarray = _fuse_embeddings(
        model, img_emb, txt_emb, device, item_encode_batch_size
    )
    logger.info("Fused embeddings: %s", fused_emb.shape)

    np.save(embeddings_path, fused_emb)
    logger.info("Embeddings saved → %s", embeddings_path)

    # ── Step 6: build FAISS index ─────────────────────────────────────────────
    logger.info("Step 5/6 — building FAISS index")
    article_ids_str = [str(i + 1) for i in range(len(items))]
    retriever = _build_retriever(fused_emb, article_ids_str)
    retriever.save(str(index_dir))
    logger.info("FAISS index saved → %s (%d items)", index_dir, len(items))

    # ── Step 7: write catalog parquet ─────────────────────────────────────────
    logger.info("Step 6/6 — writing catalog parquet")
    rows = [
        {
            "article_id": i + 1,
            "product_id": item.product_id,
            "title": item.title,
            "description": item.description,
            "category": item.category,
            "price_inr": item.price_inr,
            "pdp_url": item.pdp_url,
            "image_url": item.image_url,
        }
        for i, item in enumerate(items)
    ]
    catalog_df = pd.DataFrame(rows)
    catalog_df.to_parquet(catalog_parquet_path, index=False)
    logger.info("Catalog parquet saved → %s (%d rows)", catalog_parquet_path, len(catalog_df))

    # ── Step 8: write brand YAML ──────────────────────────────────────────────
    brand_config = {
        "brand": brand,
        "display_name": brand.replace("_", " ").title(),
        "catalog_path": f"data/{brand}/items.parquet",
        "index_path": f"indices/{brand}/active.faiss",
        "embeddings_path": f"indices/{brand}/item_emb.npy",
        "pdp_url_template": None,
        "api_key_env": f"{brand.upper()}_API_KEY",
        "llm": {"provider": "template", "enabled": True},
    }
    brand_yaml_path.write_text(
        yaml.dump(brand_config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    logger.info("Brand YAML written → %s", brand_yaml_path)

    return brand_yaml_path
