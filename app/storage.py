from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml

if TYPE_CHECKING:
    from app.brands.registry import BrandConfig

logger = structlog.get_logger(__name__)


def _gcs_client() -> Any:
    """Return a google.cloud.storage.Client. Raises ImportError if package absent."""
    from google.cloud import storage  # noqa: PLC0415

    return storage.Client(project=os.environ.get("GCS_PROJECT"))


def _download_if_missing(client: Any, bucket_name: str, local_path: str) -> bool:
    """
    Download `local_path` from GCS if it does not exist locally.
    GCS object key == local relative path (bucket mirrors repo layout).
    Returns True if a download was performed.
    """
    p = Path(local_path)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(local_path)
    blob.download_to_filename(str(p))
    logger.info("gcs_download_ok", path=local_path, bucket=bucket_name)
    return True


def brand_asset_paths(cfg: BrandConfig) -> list[str]:
    """
    Return all data file paths a single validated BrandConfig requires.

    Paths are derived from the VALIDATED BrandConfig — not the raw YAML dict — so
    pydantic DEFAULTS are honoured. This matters for `checkpoint_path`: brands that
    omit it still resolve to the default `checkpoints/best.pt` that
    registry._load_brand loads. Reading raw YAML here previously skipped that default,
    so the checkpoint was never synced and the container crashed at torch.load. Driving
    both the collector and the loader from the same BrandConfig prevents that drift.

    For `index_path` (a directory), expands to the two constituent files that
    FaissRetriever.load() expects: `{index_path}/faiss.index` and
    `{index_path}/article_ids.pkl`. `transactions_dir` expands to the three split parquets.

    This is also the single source of truth reused by `scripts/brand_preflight.py`, so a
    new brand's pre-flight local/GCS check and the runtime GCS sync can never diverge —
    this exact divergence class caused the Phase 7 "5th boot blocker" incident.
    """
    paths: list[str] = []
    for val in (cfg.catalog_path, cfg.checkpoint_path, cfg.embeddings_path):
        if val:
            paths.append(val)
    base = cfg.index_path.rstrip("/")
    paths.append(f"{base}/faiss.index")
    paths.append(f"{base}/article_ids.pkl")
    # Sync the per-brand CLIP-512 visual index when configured.
    if cfg.visual_index_path:
        vbase = cfg.visual_index_path.rstrip("/")
        paths.append(f"{vbase}/faiss.index")
        paths.append(f"{vbase}/article_ids.pkl")
    if cfg.color_index_path:
        paths.append(cfg.color_index_path)
    if cfg.attributes_path:
        paths.append(cfg.attributes_path)
    if cfg.transactions_dir:
        tbase = cfg.transactions_dir.rstrip("/")
        for split in cfg.transaction_splits:
            paths.append(f"{tbase}/{split}.parquet")
    return paths


def _collect_brand_paths(brands_dir: str) -> list[str]:
    """
    Return all data file paths referenced by brand YAML configs.

    Thin wrapper: loads + validates each enabled brand's YAML into a BrandConfig, then
    delegates per-brand path derivation to `brand_asset_paths` so this function and any
    other caller (e.g. the CLI pre-flight tool) can never diverge.

    Respects the BRANDS_ENABLED env var: when set, only paths for listed brand slugs
    are returned (matched on the yaml `brand:` field, not filename).
    """
    from app.brands.registry import (  # noqa: PLC0415 — avoid circular at module level
        BrandConfig,
        _enabled_brands,
    )

    enabled = _enabled_brands()
    paths: list[str] = []
    for yaml_path in sorted(Path(brands_dir).glob("*.yaml")):
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        if enabled is not None and str(data.get("brand", "")).lower() not in enabled:
            continue
        cfg = BrandConfig.model_validate(data)
        paths.extend(brand_asset_paths(cfg))
    return paths


def sync_brand_assets(brands_dir: str = "brands") -> None:
    """
    Download brand data files from GCS if GCS_BUCKET_NAME env var is set.
    No-op when GCS_BUCKET_NAME is absent (local dev uses local files).
    Raises on download failure — Cloud Run startup should fail loudly on missing data.
    """
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        logger.debug("gcs_sync_skipped", reason="GCS_BUCKET_NAME not set")
        return
    try:
        client = _gcs_client()
    except ImportError:
        logger.warning("gcs_sync_skipped", reason="google-cloud-storage not installed")
        return
    paths = _collect_brand_paths(brands_dir)
    downloaded = 0
    for path in paths:
        if _download_if_missing(client, bucket_name, path):
            downloaded += 1
    logger.info("gcs_sync_complete", downloaded=downloaded, total=len(paths))
