"""tests/test_visual_search_fashionclip_serve_path.py

Regression guard: T-Shirt/Shirt confusion bug over HTTP (serve path).

Background
----------
Under the CLIP-era encoder, Powerlook T-Shirt image queries frequently returned
Shirt-category results (571 Shirts vs 229 T-Shirts in the catalog meant near-exact
Shirt matches numerically dominated a T-Shirt query's top-5 — see PROJECT_MEMORY.md
Phase 9). FashionCLIP fixes this at the encoder level. This test locks the fix in
via the REAL HTTP route (FastAPI TestClient), the REAL `visual_fashionclip.faiss`
index, and the REAL FashionCLIP encoder — not a mock — so it exercises the exact
serve path a buyer's request takes.

Follows the mock-registry + TestClient pattern established in
``tests/test_visual_search_self_retrieval.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

BRAND = "powerlook"
# article_id=1: first T-Shirt in the catalog by article_id — a stable, non-arbitrary
# pick (not randomly selected). Empirically verified 5/5 top-5 results are T-Shirt
# under FashionCLIP before committing to the >=4/5 threshold below.
ARTICLE_ID = 1
TOP_K = 5


def _image_path_for_article(brand: str, article_id: int) -> Path | None:
    """Mirror of eval_visual_search._image_path: parquet product_id -> local jpg."""
    catalog = pd.read_parquet(REPO_ROOT / "data" / brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)
    rows = catalog[catalog["article_id"] == article_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    images_dir = REPO_ROOT / "data" / brand / "images"
    pid = str(row.get("product_id", "")).strip()
    if pid:
        p = images_dir / f"{pid}.jpg"
        if p.exists():
            return p
    aid_str = str(article_id)
    p = images_dir / f"{aid_str}.jpg"
    return p if p.exists() else None


def test_visual_search_tshirt_query_not_confused_with_shirt() -> None:
    """A T-Shirt image query must return mostly T-Shirt results, not Shirt.

    Regression bar: at least 4 of the top-5 results must be category=="T-Shirt".
    Not 5/5 — a stricter bar would make this test fragile to the rare single-item
    edge case already found and worked around for the snitch article_id=17 fixture
    (1 stray off-category item in an otherwise-clean index; see PROJECT_MEMORY.md
    "Migration" note). article_id=1 empirically hits 5/5 today, so >=4/5 leaves a
    one-item margin without weakening the guard against the systemic CLIP-era bug
    (which put 1-2 Shirt results in nearly every T-Shirt query's top-5).
    """
    pytest.importorskip("transformers", reason="transformers required for FashionCLIP")

    visual_index_dir = REPO_ROOT / "indices" / BRAND / "visual_fashionclip.faiss"
    if not visual_index_dir.is_dir():
        pytest.skip(f"Visual index not found: {visual_index_dir}")

    img_path = _image_path_for_article(BRAND, ARTICLE_ID)
    if img_path is None:
        pytest.skip(f"No local image for article_id={ARTICLE_ID}")

    img_bytes = img_path.read_bytes()

    from src.retrieval.faiss_index import FaissRetriever

    visual_retriever = FaissRetriever.load(str(visual_index_dir))

    catalog = pd.read_parquet(REPO_ROOT / "data" / BRAND / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)
    art_map = catalog.set_index("article_id").to_dict("index")

    from app.rerank import RerankConfig

    state = MagicMock()
    state.api_key = "vs-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.visual_retriever = visual_retriever
    state.config.rerank = RerankConfig(enabled=True, candidate_pool_size=50)
    # C1: empty color index so color_rerank is a no-op in this test.
    state.color_index = {}

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == BRAND else None
    registry.brand_names.return_value = [BRAND]

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{BRAND}/visual-search",
                files={"image": (img_path.name, img_bytes, "image/jpeg")},
                params={"k": TOP_K},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    item_ids = [r["item_id"] for r in body["results"]]
    result_cats = [
        art_map.get(int(iid), {}).get("category", "UNKNOWN") for iid in item_ids if iid.isdigit()
    ]
    tshirt_count = sum(1 for c in result_cats if c == "T-Shirt")

    assert tshirt_count >= 4, (
        f"T-Shirt/Shirt confusion regression: only {tshirt_count}/{len(result_cats)} "
        f"top-{TOP_K} results were category=='T-Shirt' for a T-Shirt query "
        f"(article_id={ARTICLE_ID}). Categories returned: {result_cats}. "
        "This is the exact CLIP-era bug FashionCLIP was migrated in to fix — "
        "check that visual_index_path/app/visual.py still point at FashionCLIP."
    )
