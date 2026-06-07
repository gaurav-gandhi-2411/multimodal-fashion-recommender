from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.ingestion.images import download_images
from app.ingestion.schema import CatalogRow


def _make_item(pid: str = "SN001", url: str = "https://cdn.example.com/SN001.jpg") -> CatalogRow:
    return CatalogRow(
        product_id=pid,
        title="Test Tee",
        description="A test item",
        image_url=url,
        price_inr=999.0,
        category="topwear",
        pdp_url=f"https://example.com/products/{pid}",
    )


def _success_response(content: bytes = b"fake_image") -> MagicMock:
    resp = MagicMock()
    resp.iter_content.return_value = [content]
    resp.raise_for_status.return_value = None
    return resp


class TestDownloadImages:
    def test_successful_download_creates_file(self, tmp_path: Path):
        item = _make_item()
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.return_value = _success_response(b"image_bytes")
            results = download_images([item], tmp_path / "images", max_workers=1)

        assert results["SN001"] is not None
        assert (tmp_path / "images" / "SN001.jpg").read_bytes() == b"image_bytes"

    def test_resume_skips_existing_file(self, tmp_path: Path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        existing = images_dir / "SN001.jpg"
        existing.write_bytes(b"original_content")

        item = _make_item()
        with patch("app.ingestion.images.requests.get") as mock_get:
            results = download_images([item], images_dir, max_workers=1)
            mock_get.assert_not_called()

        assert results["SN001"] == existing
        assert existing.read_bytes() == b"original_content"

    def test_retry_on_transient_error(self, tmp_path: Path):
        item = _make_item(pid="SN002", url="https://cdn.example.com/SN002.jpg")
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.side_effect = [
                ConnectionError("Network blip"),
                _success_response(b"img_data"),
            ]
            results = download_images(
                [item], tmp_path / "images", max_workers=1, max_retries=3, backoff_base=0.01
            )

        assert results["SN002"] is not None
        assert mock_get.call_count == 2

    def test_all_retries_exhausted_returns_none(self, tmp_path: Path):
        item = _make_item(pid="SN003")
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Always fails")
            results = download_images(
                [item], tmp_path / "images", max_workers=1, max_retries=2, backoff_base=0.01
            )

        assert results["SN003"] is None

    def test_failure_manifest_written_on_failure(self, tmp_path: Path):
        item = _make_item(pid="SN003", url="https://cdn.example.com/SN003.jpg")
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Always fails")
            download_images(
                [item], tmp_path / "images", max_workers=1, max_retries=2, backoff_base=0.01
            )

        manifest_path = tmp_path / "images" / "failed_images.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "SN003" in manifest
        assert manifest["SN003"]["url"] == "https://cdn.example.com/SN003.jpg"
        assert "error" in manifest["SN003"]

    def test_failure_manifest_is_cumulative(self, tmp_path: Path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        old_entry = {"SN_OLD": {"url": "https://cdn.example.com/old.jpg", "error": "timeout"}}
        (images_dir / "failed_images.json").write_text(json.dumps(old_entry))

        item = _make_item(pid="SN_NEW", url="https://cdn.example.com/SN_NEW.jpg")
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Network error")
            download_images(
                item if False else [item],
                images_dir,
                max_workers=1,
                max_retries=1,
                backoff_base=0.0,
            )

        updated = json.loads((images_dir / "failed_images.json").read_text())
        assert "SN_OLD" in updated
        assert "SN_NEW" in updated

    def test_success_removes_entry_from_manifest(self, tmp_path: Path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        old_manifest = {"SN001": {"url": "https://cdn.example.com/SN001.jpg", "error": "old error"}}
        (images_dir / "failed_images.json").write_text(json.dumps(old_manifest))

        item = _make_item()
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.return_value = _success_response(b"good_data")
            download_images([item], images_dir, max_workers=1)

        updated = json.loads((images_dir / "failed_images.json").read_text())
        assert "SN001" not in updated

    def test_multiple_items_all_succeed(self, tmp_path: Path):
        items = [
            _make_item(pid=f"SN{i:03d}", url=f"https://cdn.example.com/SN{i:03d}.jpg")
            for i in range(5)
        ]
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.return_value = _success_response(b"img")
            results = download_images(items, tmp_path / "images", max_workers=4)

        assert len(results) == 5
        assert all(v is not None for v in results.values())

    def test_partial_failure_mixed_results(self, tmp_path: Path):
        good = _make_item(pid="GOOD", url="https://cdn.example.com/GOOD.jpg")
        bad = _make_item(pid="BAD", url="https://cdn.example.com/BAD.jpg")

        def side_effect(url, **kwargs):
            if "GOOD" in url:
                return _success_response(b"ok")
            raise ConnectionError("fail")

        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.side_effect = side_effect
            results = download_images(
                [good, bad], tmp_path / "images", max_workers=1, max_retries=1, backoff_base=0.0
            )

        assert results["GOOD"] is not None
        assert results["BAD"] is None

    def test_extension_detection_png(self, tmp_path: Path):
        item = _make_item(pid="PNG001", url="https://cdn.example.com/PNG001.png")
        with patch("app.ingestion.images.requests.get") as mock_get:
            mock_get.return_value = _success_response(b"png_data")
            results = download_images([item], tmp_path / "images", max_workers=1)

        assert results["PNG001"] is not None
        assert (tmp_path / "images" / "PNG001.png").exists()
