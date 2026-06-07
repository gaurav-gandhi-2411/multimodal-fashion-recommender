from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.schema import CatalogRow, InteractionRow


def _good_catalog_dict() -> dict:
    return {
        "product_id": "SN001",
        "title": "Oversized Tee",
        "description": "100% cotton drop-shoulder",
        "image_url": "https://cdn.snitch.co.in/img/SN001.jpg",
        "price_inr": 1299.0,
        "category": "topwear",
        "pdp_url": "https://snitch.co.in/products/oversized-tee",
    }


def _good_interaction_dict() -> dict:
    return {
        "user_id": "u1",
        "product_id": "SN001",
        "timestamp": "2025-11-01T10:00:00Z",
        "event_type": "purchase",
    }


class TestCatalogRow:
    def test_valid_row_passes(self):
        row = CatalogRow(**_good_catalog_dict())
        assert row.product_id == "SN001"
        assert row.price_inr == 1299.0

    def test_strips_whitespace_from_strings(self):
        d = _good_catalog_dict()
        d["title"] = "  Oversized Tee  "
        row = CatalogRow(**d)
        assert row.title == "Oversized Tee"

    def test_empty_product_id_raises(self):
        d = _good_catalog_dict()
        d["product_id"] = ""
        with pytest.raises(ValidationError, match="empty"):
            CatalogRow(**d)

    def test_whitespace_title_raises(self):
        d = _good_catalog_dict()
        d["title"] = "   "
        with pytest.raises(ValidationError, match="empty"):
            CatalogRow(**d)

    def test_empty_description_raises(self):
        d = _good_catalog_dict()
        d["description"] = ""
        with pytest.raises(ValidationError, match="empty"):
            CatalogRow(**d)

    def test_negative_price_raises(self):
        d = _good_catalog_dict()
        d["price_inr"] = -1.0
        with pytest.raises(ValidationError, match="positive"):
            CatalogRow(**d)

    def test_zero_price_raises(self):
        d = _good_catalog_dict()
        d["price_inr"] = 0.0
        with pytest.raises(ValidationError, match="positive"):
            CatalogRow(**d)

    def test_missing_required_field_raises(self):
        d = _good_catalog_dict()
        del d["category"]
        with pytest.raises(ValidationError):
            CatalogRow(**d)

    def test_empty_image_url_raises(self):
        d = _good_catalog_dict()
        d["image_url"] = ""
        with pytest.raises(ValidationError, match="empty"):
            CatalogRow(**d)

    def test_empty_pdp_url_raises(self):
        d = _good_catalog_dict()
        d["pdp_url"] = ""
        with pytest.raises(ValidationError, match="empty"):
            CatalogRow(**d)


class TestInteractionRow:
    def test_valid_row_passes(self):
        row = InteractionRow(**_good_interaction_dict())
        assert row.user_id == "u1"
        assert row.event_type == "purchase"

    def test_all_valid_event_types(self):
        for et in ("view", "purchase", "wishlist", "cart"):
            d = _good_interaction_dict()
            d["event_type"] = et
            row = InteractionRow(**d)
            assert row.event_type == et

    def test_invalid_event_type_raises(self):
        d = _good_interaction_dict()
        d["event_type"] = "click"
        with pytest.raises(ValidationError, match="not valid"):
            InteractionRow(**d)

    def test_empty_user_id_raises(self):
        d = _good_interaction_dict()
        d["user_id"] = ""
        with pytest.raises(ValidationError, match="empty"):
            InteractionRow(**d)

    def test_empty_product_id_raises(self):
        d = _good_interaction_dict()
        d["product_id"] = ""
        with pytest.raises(ValidationError, match="empty"):
            InteractionRow(**d)

    def test_timestamp_parsed_from_iso_string(self):
        row = InteractionRow(**_good_interaction_dict())
        assert row.timestamp.year == 2025

    def test_missing_timestamp_raises(self):
        d = _good_interaction_dict()
        del d["timestamp"]
        with pytest.raises(ValidationError):
            InteractionRow(**d)
