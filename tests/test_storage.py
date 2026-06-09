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
    """_collect_brand_paths extracts all file paths from brand YAML."""
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
    assert "indices/test/active.faiss" in paths
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
    index = str(assets_root / "active.faiss")
    checkpoint = str(assets_root / "best.pt")
    yaml_content = f"""
brand: testbrand
display_name: Test
catalog_path: {catalog}
index_path: {index}
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

    # download_to_filename called for each non-existing path
    assert mock_blob.download_to_filename.call_count == 3  # catalog, index, checkpoint
