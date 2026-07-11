from __future__ import annotations

from app.ingestion.category_normalize import canonicalize_category


class TestCanonicalizeCategory:
    def test_fashor_dress_maps_to_dresses(self):
        assert canonicalize_category("fashor", "Dress") == "Dresses"

    def test_fashor_kurta_maps_to_kurtas(self):
        assert canonicalize_category("fashor", "Kurta") == "Kurtas"

    def test_fashor_kurta_set_is_not_merged(self):
        # "Kurta Set" is a genuinely different product (multi-piece set) -- must stay distinct.
        assert canonicalize_category("fashor", "Kurta Set") == "Kurta Set"

    def test_fashor_dresses_is_already_canonical(self):
        assert canonicalize_category("fashor", "Dresses") == "Dresses"

    def test_virgio_skort_maps_to_skorts(self):
        assert canonicalize_category("virgio", "Skort") == "Skorts"

    def test_unregistered_brand_is_a_noop(self):
        assert canonicalize_category("snitch", "Shirts") == "Shirts"

    def test_unregistered_category_is_a_noop(self):
        assert canonicalize_category("fashor", "Bottom") == "Bottom"
