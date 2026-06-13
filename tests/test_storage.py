from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.storage import _collect_brand_paths, _download_if_missing, sync_brand_assets


def test_sync_brand_assets_no_op_when_no_bucket_env(tmp_path: Path) -> None:
    """sync_brand_assets must be a no-op when GCS_BUCKET_NAME is not set."""
    env = {k: v for k, v in os.environ.items() if k != "GCS_BUCKET_NAME"}
    with patch.dict(os.environ, env, clear=True):
        # Should not raise, should not import google.cloud.storage
        sync_brand_assets(brands_dir=str(tmp_path))


def test_collect_brand_paths_returns_all_fields(tmp_path: Path) -> None:
    """_collect_brand_paths extracts all file paths from brand YAML.

    index_path is a directory; it must be expanded into the two files that
    FaissRetriever.load() reads: faiss.index and article_ids.pkl.
    The bare directory path must NOT appear in the results.
    """
    yaml_content = """
brand: testbrand
display_name: Test Brand
catalog_path: data/test/catalog.parquet
index_path: indices/test/active.faiss
checkpoint_path: checkpoints/best.pt
transactions_dir: data/test/transactions/
api_key_env: TEST_API_KEY
"""
    (tmp_path / "testbrand.yaml").write_text(yaml_content)
    paths = _collect_brand_paths(str(tmp_path))
    assert "data/test/catalog.parquet" in paths
    # index_path expands to two constituent files, not the bare directory
    assert "indices/test/active.faiss/faiss.index" in paths
    assert "indices/test/active.faiss/article_ids.pkl" in paths
    assert "indices/test/active.faiss" not in paths
    assert "checkpoints/best.pt" in paths
    assert "data/test/transactions/train.parquet" in paths
    assert "data/test/transactions/val.parquet" in paths
    assert "data/test/transactions/test.parquet" in paths


def test_download_if_missing_skips_existing_file(tmp_path: Path) -> None:
    """_download_if_missing returns False when file already exists locally."""
    existing = tmp_path / "data" / "file.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_text("content")
    mock_client = MagicMock()
    result = _download_if_missing(mock_client, "my-bucket", str(existing))
    assert result is False
    mock_client.bucket.assert_not_called()


def test_download_if_missing_downloads_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_download_if_missing calls blob.download_to_filename for a missing file."""
    local_path = str(tmp_path / "new_dir" / "file.faiss")
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    result = _download_if_missing(mock_client, "my-bucket", local_path)

    assert result is True
    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with(local_path)
    mock_blob.download_to_filename.assert_called_once_with(local_path)
    assert Path(local_path).parent.exists()


def test_sync_brand_assets_calls_gcs_when_bucket_set(tmp_path: Path) -> None:
    """sync_brand_assets calls GCS download for each asset when GCS_BUCKET_NAME is set."""
    # Use tmp_path-rooted asset paths so they are guaranteed absent on disk.
    # Relative paths like "checkpoints/best.pt" can collide with real project files
    # when pytest runs from the repo root.
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    assets_root = tmp_path / "assets"
    catalog = str(assets_root / "catalog.parquet")
    # index_path is a directory; storage.py expands it to faiss.index + article_ids.pkl
    index_dir = str(assets_root / "active.faiss")
    checkpoint = str(assets_root / "best.pt")
    yaml_content = f"""
brand: testbrand
display_name: Test
catalog_path: {catalog}
index_path: {index_dir}
checkpoint_path: {checkpoint}
api_key_env: TEST_KEY
"""
    (brands_dir / "testbrand.yaml").write_text(yaml_content)
    mock_client = MagicMock()
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    with (
        patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}, clear=False),
        patch("app.storage._gcs_client", return_value=mock_client),
    ):
        sync_brand_assets(brands_dir=str(brands_dir))

    # download_to_filename called for each non-existing path:
    # catalog, active.faiss/faiss.index, active.faiss/article_ids.pkl, checkpoint = 4
    assert mock_blob.download_to_filename.call_count == 4


def _write_brand_yaml(brands_dir: Path, slug: str) -> None:
    """Helper: write a minimal brand YAML with the given slug into brands_dir."""
    content = f"""
brand: {slug}
display_name: {slug.title()}
catalog_path: data/{slug}/catalog.parquet
index_path: indices/{slug}/active.faiss
checkpoint_path: checkpoints/{slug}/best.pt
api_key_env: {slug.upper()}_API_KEY
"""
    (brands_dir / f"{slug}.yaml").write_text(content)


def test_collect_brand_paths_brands_enabled_filter(tmp_path: Path) -> None:
    """_collect_brand_paths respects BRANDS_ENABLED: only enabled brand paths returned."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    _write_brand_yaml(brands_dir, "alpha")
    _write_brand_yaml(brands_dir, "beta")

    # With BRANDS_ENABLED set to "alpha" only, beta paths must be absent.
    with patch.dict(os.environ, {"BRANDS_ENABLED": "alpha"}, clear=False):
        paths = _collect_brand_paths(str(brands_dir))

    alpha_paths = [p for p in paths if "alpha" in p]
    beta_paths = [p for p in paths if "beta" in p]
    assert alpha_paths, "alpha paths must be present when alpha is enabled"
    assert not beta_paths, "beta paths must be absent when beta is not enabled"

    # With BRANDS_ENABLED unset, both brands' paths must appear.
    env_without = {k: v for k, v in os.environ.items() if k != "BRANDS_ENABLED"}
    with patch.dict(os.environ, env_without, clear=True):
        all_paths = _collect_brand_paths(str(brands_dir))

    assert any("alpha" in p for p in all_paths), "alpha paths expected when filter unset"
    assert any("beta" in p for p in all_paths), "beta paths expected when filter unset"
