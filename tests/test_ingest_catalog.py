from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.schema import CatalogRow
from app.ingestion.sources import CsvSource, ShopifySource


# ── helpers ──────────────────────────────────────────────────────────────────

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "snitch_catalog.csv"


def _make_shopify_product(pid: int, title: str = "Test Tee", price: str = "999") -> dict:
    return {
        "id": pid,
        "title": title,
        "handle": f"test-tee-{pid}",
        "product_type": "topwear",
        "body_html": "<p>Nice shirt</p>",
        "vendor": "TestBrand",
        "variants": [{"price": price}],
        "images": [{"src": f"https://cdn.example.com/{pid}.jpg"}],
    }


def _mock_response(json_data=None, status_code: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


# ── CsvSource ─────────────────────────────────────────────────────────────────

class TestCsvSource:
    def test_loads_fixture_csv(self):
        source = CsvSource(FIXTURE_CSV)
        rows = source.fetch()
        assert len(rows) == 20
        assert all(isinstance(r, CatalogRow) for r in rows)

    def test_first_row_values(self):
        source = CsvSource(FIXTURE_CSV)
        rows = source.fetch()
        assert rows[0].product_id == "SN001"
        assert rows[0].price_inr == 1299.0
        assert rows[0].category == "topwear"

    def test_missing_required_column_raises(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("product_id,title,description,image_url,price_inr,pdp_url\nSN001,Tee,Desc,http://x.com/a.jpg,999,http://x.com/p\n")
        source = CsvSource(bad_csv)
        with pytest.raises(ValueError, match="missing required columns"):
            source.fetch()

    def test_missing_column_error_names_the_column(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("product_id,title,description,image_url,price_inr,pdp_url\nSN001,Tee,Desc,http://x.com/a.jpg,999,http://x.com/p\n")
        source = CsvSource(bad_csv)
        with pytest.raises(ValueError, match="category"):
            source.fetch()

    def test_invalid_price_row_skipped_with_warning(self, tmp_path: Path):
        csv_content = (
            "product_id,title,description,image_url,price_inr,category,pdp_url\n"
            "SN001,Tee,Desc,http://x.com/a.jpg,-100,topwear,http://x.com/p\n"  # bad
            "SN002,Shirt,Desc2,http://x.com/b.jpg,999,topwear,http://x.com/q\n"  # good
        )
        (tmp_path / "catalog.csv").write_text(csv_content)
        source = CsvSource(tmp_path / "catalog.csv")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rows = source.fetch()
        assert len(rows) == 1
        assert rows[0].product_id == "SN002"
        assert len(w) == 1
        assert "invalid" in str(w[0].message).lower()

    def test_majority_invalid_raises(self, tmp_path: Path):
        # All rows bad (price=0)
        lines = ["product_id,title,description,image_url,price_inr,category,pdp_url"]
        for i in range(10):
            lines.append(f"SN{i:03d},Tee,Desc,http://x.com/{i}.jpg,0,topwear,http://x.com/p{i}")
        (tmp_path / "catalog.csv").write_text("\n".join(lines) + "\n")
        source = CsvSource(tmp_path / "catalog.csv")
        with pytest.raises(ValueError, match="Too many invalid"):
            source.fetch()


# ── ShopifySource ─────────────────────────────────────────────────────────────

class TestShopifySource:
    BASE_URL = "https://testbrand.myshopify.com"
    PRODUCTS_URL = f"{BASE_URL}/products.json"

    def _make_source(self, respect_robots: bool = False) -> ShopifySource:
        return ShopifySource(self.PRODUCTS_URL, respect_robots=respect_robots)

    def _make_session_mock(self, responses: list) -> MagicMock:
        """Return a mock Session whose .get() returns responses in order."""
        session = MagicMock()
        session.headers = {}
        session.get.side_effect = responses
        return session

    def test_fetch_two_pages_mocked(self):
        source = self._make_source()

        robots_resp = _mock_response(text="User-agent: *\nAllow: /")
        probe_resp = _mock_response(json_data={"products": [_make_shopify_product(1)]})
        page1_prods = [_make_shopify_product(i) for i in range(1, 251)]  # 250 items
        page1_resp = _mock_response(json_data={"products": page1_prods})
        page2_prods = [_make_shopify_product(251)]  # 1 item → last page
        page2_resp = _mock_response(json_data={"products": page2_prods})
        page3_resp = _mock_response(json_data={"products": []})  # empty

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock(
                [robots_resp, probe_resp, page1_resp, page2_resp, page3_resp]
            )
            rows = source.fetch()

        assert len(rows) == 251

    def test_fetch_normalizes_html_description(self):
        source = self._make_source()
        p = _make_shopify_product(1)
        p["body_html"] = "<p>Nice &amp; comfy shirt</p>"

        robots_resp = _mock_response(text="")
        probe_resp = _mock_response(json_data={"products": [p]})
        page1_resp = _mock_response(json_data={"products": [p]})
        page2_resp = _mock_response(json_data={"products": []})

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock(
                [robots_resp, probe_resp, page1_resp, page2_resp]
            )
            rows = source.fetch()

        assert "Nice & comfy shirt" in rows[0].description

    def test_404_raises_with_csv_hint(self):
        source = self._make_source()
        robots_resp = _mock_response(text="")
        probe_resp = _mock_response(status_code=404)

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock([robots_resp, probe_resp])
            with pytest.raises(RuntimeError, match="404"):
                source.fetch()

    def test_404_error_mentions_csv_path(self):
        source = self._make_source()
        robots_resp = _mock_response(text="")
        probe_resp = _mock_response(status_code=404)

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock([robots_resp, probe_resp])
            with pytest.raises(RuntimeError, match="csv"):
                source.fetch()

    def test_non_shopify_json_raises(self):
        source = self._make_source()
        robots_resp = _mock_response(text="")
        probe_resp = _mock_response(json_data={"items": []})  # no "products" key

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock([robots_resp, probe_resp])
            with pytest.raises(RuntimeError, match="Shopify JSON"):
                source.fetch()

    def test_robots_disallow_warns_by_default(self, caplog):
        source = self._make_source(respect_robots=False)

        robots_text = "User-agent: *\nDisallow: /products.json\n"
        robots_resp = _mock_response(text=robots_text)
        probe_resp = _mock_response(json_data={"products": [_make_shopify_product(1)]})
        page1_resp = _mock_response(json_data={"products": [_make_shopify_product(1)]})
        page2_resp = _mock_response(json_data={"products": []})

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock(
                [robots_resp, probe_resp, page1_resp, page2_resp]
            )
            import logging
            with caplog.at_level(logging.WARNING, logger="app.ingestion.sources"):
                rows = source.fetch()

        # Proceeds successfully despite robots.txt
        assert len(rows) == 1
        assert "robots.txt" in caplog.text.lower()

    def test_robots_disallow_raises_with_respect_flag(self):
        source = self._make_source(respect_robots=True)

        robots_text = "User-agent: *\nDisallow: /products.json\n"
        robots_resp = _mock_response(text=robots_text)

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock([robots_resp])
            with pytest.raises(RuntimeError, match="disallows"):
                source.fetch()

    def test_products_with_zero_price_skipped(self):
        source = self._make_source()
        bad = _make_shopify_product(99, price="0")
        good = _make_shopify_product(100, price="1299")

        robots_resp = _mock_response(text="")
        probe_resp = _mock_response(json_data={"products": [good]})
        page1_resp = _mock_response(json_data={"products": [bad, good]})
        page2_resp = _mock_response(json_data={"products": []})

        with patch("app.ingestion.sources.requests.Session") as MockSession:
            MockSession.return_value = self._make_session_mock(
                [robots_resp, probe_resp, page1_resp, page2_resp]
            )
            rows = source.fetch()

        assert len(rows) == 1
        assert rows[0].product_id == "100"


# ── Pipeline integration tests ────────────────────────────────────────────────

import numpy as np

from app.ingestion.pipeline import run_catalog_pipeline


@pytest.fixture
def mock_pipeline_ml():
    """Patch ML helpers in pipeline so integration tests run without GPU/CLIP/SBERT."""

    def _fake_download(items, images_dir, **kwargs):
        images_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for item in items:
            p = images_dir / f"{item.product_id}.jpg"
            p.write_bytes(b"x")
            result[item.product_id] = p
        return result

    def _fake_build_img(cfg):
        enc = MagicMock()

        def _encode(paths):
            n = len(paths)
            e = np.random.randn(n, 512).astype(np.float32)
            return e / np.linalg.norm(e, axis=1, keepdims=True)

        enc.encode_batch.side_effect = _encode
        return enc

    def _fake_build_txt(cfg):
        enc = MagicMock()

        def _encode(texts):
            n = len(texts)
            e = np.random.randn(n, 384).astype(np.float32)
            return e / np.linalg.norm(e, axis=1, keepdims=True)

        enc.encode_batch.side_effect = _encode
        return enc

    def _fake_load_model(checkpoint_path, device):
        return MagicMock()

    def _fake_fuse(model, img_emb, txt_emb, device, batch_size):
        n = len(img_emb)
        e = np.random.randn(n, 256).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    with (
        patch("app.ingestion.pipeline.download_images", side_effect=_fake_download),
        patch("app.ingestion.pipeline._build_img_encoder", side_effect=_fake_build_img),
        patch("app.ingestion.pipeline._build_txt_encoder", side_effect=_fake_build_txt),
        patch("app.ingestion.pipeline._load_two_tower", side_effect=_fake_load_model),
        patch("app.ingestion.pipeline._fuse_embeddings", side_effect=_fake_fuse),
    ):
        yield


class TestPipelineIntegration:
    def test_csv_pipeline_produces_all_outputs(self, tmp_path: Path, mock_pipeline_ml):
        source = CsvSource(FIXTURE_CSV)
        items = source.fetch()

        yaml_path = run_catalog_pipeline(items, "snitch", output_base=tmp_path)

        assert yaml_path.exists()
        assert yaml_path == tmp_path / "brands" / "snitch.yaml"
        assert (tmp_path / "data" / "snitch" / "items.parquet").exists()

        index_dir = tmp_path / "indices" / "snitch" / "active.faiss"
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "article_ids.pkl").exists()
        assert (tmp_path / "indices" / "snitch" / "item_emb.npy").exists()

    def test_faiss_index_has_correct_item_count(self, tmp_path: Path, mock_pipeline_ml):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.retrieval.faiss_index import FaissRetriever

        source = CsvSource(FIXTURE_CSV)
        items = source.fetch()
        run_catalog_pipeline(items, "snitch", output_base=tmp_path)

        index_dir = tmp_path / "indices" / "snitch" / "active.faiss"
        retriever = FaissRetriever.load(str(index_dir))
        assert retriever.index.ntotal == 20
        assert retriever.article_ids[0] == "1"
        assert retriever.article_ids[-1] == "20"

    def test_parquet_preserves_original_product_ids(self, tmp_path: Path, mock_pipeline_ml):
        import pandas as pd

        source = CsvSource(FIXTURE_CSV)
        items = source.fetch()
        run_catalog_pipeline(items, "snitch", output_base=tmp_path)

        df = pd.read_parquet(tmp_path / "data" / "snitch" / "items.parquet")
        assert "article_id" in df.columns
        assert "product_id" in df.columns
        assert len(df) == 20
        assert df["product_id"].iloc[0] == "SN001"
        assert df["article_id"].iloc[0] == 1

    def test_brand_yaml_has_correct_fields(self, tmp_path: Path, mock_pipeline_ml):
        import yaml as _yaml

        source = CsvSource(FIXTURE_CSV)
        items = source.fetch()
        yaml_path = run_catalog_pipeline(items, "snitch", output_base=tmp_path)

        cfg = _yaml.safe_load(yaml_path.read_text())
        assert cfg["brand"] == "snitch"
        assert cfg["api_key_env"] == "SNITCH_API_KEY"
        assert cfg["catalog_path"] == "data/snitch/items.parquet"
        assert cfg["index_path"] == "indices/snitch/active.faiss"
        assert cfg["llm"]["provider"] == "template"

    def test_pipeline_is_idempotent(self, tmp_path: Path, mock_pipeline_ml):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.retrieval.faiss_index import FaissRetriever

        source = CsvSource(FIXTURE_CSV)
        items = source.fetch()

        run_catalog_pipeline(items, "snitch", output_base=tmp_path)
        run_catalog_pipeline(items, "snitch", output_base=tmp_path)

        index_dir = tmp_path / "indices" / "snitch" / "active.faiss"
        retriever = FaissRetriever.load(str(index_dir))
        assert retriever.index.ntotal == 20  # not 40
