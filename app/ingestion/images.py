from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.ingestion.schema import CatalogRow

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8192


def _guess_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _download_one(
    product_id: str,
    url: str,
    dest: Path,
    *,
    max_retries: int,
    backoff_base: float,
    timeout_s: float,
) -> tuple[str, Path | None, str | None]:
    """Download one image. Returns (product_id, local_path_or_None, error_or_None)."""
    if dest.exists():
        return product_id, dest, None

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".tmp")

    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout_s, stream=True)
            resp.raise_for_status()
            with open(tmp_dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
            tmp_dest.rename(dest)
            return product_id, dest, None
        except Exception as exc:
            last_error = str(exc)
            if tmp_dest.exists():
                tmp_dest.unlink(missing_ok=True)
            if attempt < max_retries - 1:
                sleep_s = backoff_base * (2**attempt)
                logger.debug(
                    "Image download attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    product_id,
                    last_error,
                    sleep_s,
                )
                time.sleep(sleep_s)

    logger.warning(
        "Image download failed for %s after %d attempts: %s",
        product_id,
        max_retries,
        last_error,
    )
    return product_id, None, last_error


def download_images(
    items: list[CatalogRow],
    images_dir: Path,
    *,
    max_workers: int = 8,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    timeout_s: float = 10.0,
) -> dict[str, Path | None]:
    """
    Download catalog item images to images_dir.

    Resumable: items whose destination file already exists are skipped.
    Retries each failed download up to max_retries times with exponential backoff.
    Writes / updates images_dir/failed_images.json with items that ultimately fail.
    Successfully downloaded items are removed from the manifest if they were previously listed.

    Returns a dict mapping product_id -> local Path (or None if download failed).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = images_dir / "failed_images.json"

    results: dict[str, Path | None] = {}
    failures: dict[str, dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_item = {
            pool.submit(
                _download_one,
                item.product_id,
                item.image_url,
                images_dir / f"{item.product_id}{_guess_extension(item.image_url)}",
                max_retries=max_retries,
                backoff_base=backoff_base,
                timeout_s=timeout_s,
            ): item
            for item in items
        }

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            product_id, path, error = future.result()
            results[product_id] = path
            if path is None:
                failures[product_id] = {
                    "url": item.image_url,
                    "error": error or "unknown",
                }

    # Merge with existing manifest, removing newly-succeeded items
    existing_manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_manifest = {}

    for pid, path in results.items():
        if path is not None:
            existing_manifest.pop(pid, None)

    existing_manifest.update(failures)

    if existing_manifest or manifest_path.exists():
        manifest_path.write_text(json.dumps(existing_manifest, indent=2))

    succeeded = sum(1 for p in results.values() if p is not None)
    logger.info(
        "Image download complete: %d succeeded, %d failed (see %s)",
        succeeded,
        len(failures),
        manifest_path,
    )
    return results
