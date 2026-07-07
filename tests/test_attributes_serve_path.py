"""tests/test_attributes_serve_path.py

Serve-path regression guard: GET /v1/{brand}/item/{item_id}/attributes over the REAL
HTTP route (FastAPI TestClient), reading the REAL data/snitch/attributes.json built by
scripts/extract_attributes.py and the REAL data/snitch/items.parquet catalog -- not
mocked -- so it exercises the exact request a caller sends.

This is also a regression guard for the attribute-reliability honesty tiers
(app.attributes.ATTRIBUTE_RELIABILITY): the eval passes documented there found color
the only category that clearly beats a naive baseline (64.6% pooled text-cross-
validation accuracy, ~90% in a manual 20-image spot-check), while pattern/fabric/
occasion are experimental (fabric ~19.7% vs ~7.7% random; occasion WORSE than a
majority-class baseline for 2 of 3 brands). The 200-path test below asserts
reliability["color"] == "validated" and the other three are "experimental" so that a
future change quietly re-labeling an unvalidated attribute as trustworthy fails CI
instead of shipping silently.

Follows the mock-registry + TestClient pattern established in
``tests/test_visual_search_fashionclip_serve_path.py`` (real data, real route, a
MagicMock only stands in for the BrandRegistry wiring so no live API key / GCS
sync is required to run this in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.attributes import load_attribute_index  # noqa: E402

BRAND = "snitch"
# article_id=1: present in data/snitch/attributes.json (verified: color=white,
# pattern=textured, fabric=linen, occasion=casual) -- a stable, non-arbitrary pick,
# same convention as ARTICLE_ID=1 in test_visual_search_fashionclip_serve_path.py.
ARTICLE_ID = 1


def _real_art_map(brand: str) -> dict[int, dict]:
    """Load the real catalog and index by article_id, mirroring app.brands.registry."""
    catalog = pd.read_parquet(REPO_ROOT / "data" / brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)
    return catalog.set_index("article_id").to_dict("index")


def _make_registry(state: MagicMock, brand: str) -> MagicMock:
    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == brand else None
    registry.brand_names.return_value = [brand]
    return registry


def test_item_attributes_200_real_data_has_reliability_tiers() -> None:
    """A real snitch item returns 200 with all 4 attributes and the honesty tiers.

    Pins reliability["color"] == "validated" and the other three == "experimental" --
    this is the regression guard: it fails if someone quietly relabels an unvalidated
    attribute as trustworthy without rerunning the evals.
    """
    attributes_path = REPO_ROOT / "data" / BRAND / "attributes.json"
    if not attributes_path.exists():
        pytest.skip(f"No attributes.json for {BRAND}: {attributes_path}")

    art_map = _real_art_map(BRAND)
    if ARTICLE_ID not in art_map:
        pytest.skip(f"article_id={ARTICLE_ID} not in {BRAND} catalog")

    attributes = load_attribute_index(attributes_path)
    if str(ARTICLE_ID) not in attributes:
        pytest.skip(f"article_id={ARTICLE_ID} has no attribute tags in {attributes_path}")

    state = MagicMock()
    state.api_key = "attrs-serve-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.attributes = attributes

    registry = _make_registry(state, BRAND)

    with patch("app.api.main.load_registry", return_value=registry):
        from fastapi.testclient import TestClient

        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{BRAND}/item/{ARTICLE_ID}/attributes",
                headers={"X-Api-Key": "attrs-serve-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["item_id"] == str(ARTICLE_ID)
    assert body["brand"] == BRAND
    for category in ("color", "pattern", "fabric", "occasion"):
        assert category in body
        assert f"{category}_confidence" in body

    assert "reliability" in body
    assert set(body["reliability"].keys()) == {"color", "pattern", "fabric", "occasion"}
    assert body["reliability"]["color"] == "validated"
    assert body["reliability"]["pattern"] == "experimental"
    assert body["reliability"]["fabric"] == "experimental"
    assert body["reliability"]["occasion"] == "experimental"


def test_item_attributes_404_unknown_item_id() -> None:
    """An item_id absent from the catalog returns HTTP 404 (not 500 or empty 200)."""
    attributes_path = REPO_ROOT / "data" / BRAND / "attributes.json"
    if not attributes_path.exists():
        pytest.skip(f"No attributes.json for {BRAND}: {attributes_path}")

    art_map = _real_art_map(BRAND)
    attributes = load_attribute_index(attributes_path)
    unknown_id = max(art_map.keys()) + 1_000_000

    state = MagicMock()
    state.api_key = "attrs-serve-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.attributes = attributes

    registry = _make_registry(state, BRAND)

    with patch("app.api.main.load_registry", return_value=registry):
        from fastapi.testclient import TestClient

        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{BRAND}/item/{unknown_id}/attributes",
                headers={"X-Api-Key": "attrs-serve-test-key"},
            )

    assert resp.status_code == 404, resp.text


def test_item_attributes_503_brand_not_configured() -> None:
    """A brand with no attribute index built at all returns HTTP 503, not 500.

    Mirrors the MagicMock-state 503 pattern used for the analogous
    /visual-search unconfigured-brand case in tests/test_visual.py
    (test_visual_search_route_503_when_no_visual_retriever): state.attributes is an
    empty dict, simulating attributes_path unset or scripts/extract_attributes.py never
    having been run for that brand.
    """
    art_map = _real_art_map(BRAND)

    state = MagicMock()
    state.api_key = "attrs-serve-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.attributes = {}  # unconfigured -- no attributes.json built for this brand

    registry = _make_registry(state, BRAND)

    with patch("app.api.main.load_registry", return_value=registry):
        from fastapi.testclient import TestClient

        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/v1/{BRAND}/item/{ARTICLE_ID}/attributes",
                headers={"X-Api-Key": "attrs-serve-test-key"},
            )

    assert resp.status_code == 503, resp.text
