"""tests/test_visual_search_self_retrieval.py

HTTP-path self-retrieval test for /visual-search.

Root-cause note
---------------
The smoke test that originally failed self-retrieval uploaded image file
``8715959763106.jpg`` (from the catalog CSV's product_id column), but the
items.parquet stores a different product_id (``9242086670498``) for that same
article_id.  The visual FAISS index is built from the PARQUET's product_id,
not the CSV.  Uploading the wrong file gives a mismatched embedding and no
self-retrieval — not a code bug, but a test-harness gap the eval script
avoids via ``_image_path(brand, row)`` which always reads from the parquet.

This test closes that gap: it uses the same image-resolution logic as
``eval_visual_search.py`` and posts the correct bytes through the REAL HTTP
route (via FastAPI TestClient), asserting the article shows up at rank 1.

Two test variants:
  1. ``test_visual_search_self_retrieval_rank1`` — uses the REAL visual FAISS
     index + the REAL CLIP encoder (open_clip required; skipped otherwise).
     Mocks only the brand registry so the TwoTower model needn't be loaded.
     This proves eval-path ≡ serve-path for the same image bytes.

  2. ``test_visual_search_self_retrieval_with_item_id_rerank`` — same as (1)
     but also passes ``?item_id=17`` to exercise the new category-coherence
     reranker.  Asserts article 17 still ranks first AND that all returned
     items share the same category as article 17.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

BRAND = "snitch"
ARTICLE_ID = 17          # known-good item with a local image
TOP_K = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_path_for_article(brand: str, article_id: int) -> Path | None:
    """Mirror of eval_visual_search._image_path: parquet product_id → local jpg."""
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


def _build_mock_state(art_map: dict, visual_retriever) -> MagicMock:
    """Minimal BrandState mock: real visual_retriever, no TwoTower model."""
    from app.rerank import RerankConfig
    state = MagicMock()
    state.api_key = "vs-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.visual_retriever = visual_retriever
    state.config.rerank = RerankConfig(enabled=True, candidate_pool_size=50)
    # C3: no min-score check in self-retrieval tests (catalogue images score near 1.0).
    state.config.visual_search_min_score = None
    # C1: empty color index so color_rerank is a no-op in these tests.
    state.color_index = {}
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_visual_search_self_retrieval_rank1() -> None:
    """Uploading an item's OWN catalog image must return that item at rank 1.

    Uses the REAL visual FAISS index and the REAL encode_query_image path
    (identical to what the serve route uses).  Requires open_clip.

    This is the HTTP-path equivalent of eval_visual_search.py's
    self_retrieval_rate metric — it goes through FastAPI's TestClient so it
    exercises the full route handler, not just FAISS + encode functions.
    """
    pytest.importorskip("open_clip", reason="open_clip required for real CLIP encoding")

    visual_index_dir = REPO_ROOT / "indices" / BRAND / "visual.faiss"
    if not visual_index_dir.is_dir():
        pytest.skip(f"Visual index not found: {visual_index_dir}")

    img_path = _image_path_for_article(BRAND, ARTICLE_ID)
    if img_path is None:
        pytest.skip(f"No local image for article_id={ARTICLE_ID}")

    img_bytes = img_path.read_bytes()

    # Load REAL visual FAISS index (don't load TwoTower — not needed).
    from src.retrieval.faiss_index import FaissRetriever
    visual_retriever = FaissRetriever.load(str(visual_index_dir))

    catalog = pd.read_parquet(REPO_ROOT / "data" / BRAND / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)
    art_map = catalog.set_index("article_id").to_dict("index")

    state = _build_mock_state(art_map, visual_retriever)
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

    assert str(ARTICLE_ID) in item_ids, (
        f"Self-retrieval FAILED: article_id={ARTICLE_ID} not in top-{TOP_K}. "
        f"Got: {item_ids}. "
        f"Image used: {img_path}. "
        "This means the encode_query_image path on the serve route diverges from "
        "the build path — check device mismatch or image-path resolution."
    )
    assert item_ids[0] == str(ARTICLE_ID), (
        f"Self-retrieval rank wrong: expected rank-1, got rank-{item_ids.index(str(ARTICLE_ID))+1}. "
        f"Results: {item_ids}"
    )


def test_visual_search_self_retrieval_with_item_id_rerank() -> None:
    """With ?item_id= the reranker applies category coherence; self-item still rank-1.

    Also verifies all returned items share the query item's category, confirming
    the category scatter bug is fixed when item_id is provided.
    """
    pytest.importorskip("open_clip", reason="open_clip required for real CLIP encoding")

    visual_index_dir = REPO_ROOT / "indices" / BRAND / "visual.faiss"
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

    state = _build_mock_state(art_map, visual_retriever)
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
                params={"k": TOP_K, "item_id": str(ARTICLE_ID)},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    item_ids = [r["item_id"] for r in body["results"]]

    assert str(ARTICLE_ID) in item_ids, (
        f"Self-retrieval with rerank FAILED: article_id={ARTICLE_ID} not in top-{TOP_K}. "
        f"Got: {item_ids}"
    )
    assert item_ids[0] == str(ARTICLE_ID), (
        f"Self-retrieval rank with rerank wrong: expected rank-1, "
        f"got rank-{item_ids.index(str(ARTICLE_ID))+1}. Results: {item_ids}"
    )

    # Category coherence: with item_id reranking, all results should share the
    # query item's category (no cross-category scatter).
    query_cat = art_map[ARTICLE_ID].get("category", "")
    result_cats = [
        art_map.get(int(iid), {}).get("category", "UNKNOWN")
        for iid in item_ids
        if iid.isdigit()
    ]
    off_cat = [c for c in result_cats if c != query_cat]
    assert not off_cat, (
        f"Category scatter with item_id rerank: expected all '{query_cat}', "
        f"got other categories: {off_cat}"
    )


def test_visual_search_pure_image_category_coherence() -> None:
    """Pure-image path (NO item_id): category inferred from rank-1 match must filter scatter.

    Before the fix: uploading a shirt image without ?item_id returned overshirts and
    jackets because the reranker only activated when item_id was provided.

    After the fix: the route infers query_cat from the rank-1 FAISS result and applies
    the same category_affinity + price reranker, so a shirt photo returns shirts.
    This tests the serve path a real buyer uses — no item_id, just an uploaded image.
    """
    pytest.importorskip("open_clip", reason="open_clip required for real CLIP encoding")

    visual_index_dir = REPO_ROOT / "indices" / BRAND / "visual.faiss"
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

    state = _build_mock_state(art_map, visual_retriever)
    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == BRAND else None
    registry.brand_names.return_value = [BRAND]

    with patch("app.api.main.load_registry", return_value=registry):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            # NO item_id — pure image upload, the path a real buyer takes
            resp = client.post(
                f"/v1/{BRAND}/visual-search",
                files={"image": (img_path.name, img_bytes, "image/jpeg")},
                params={"k": TOP_K},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    item_ids = [r["item_id"] for r in body["results"]]

    # Rank-1 must still be self-retrieval (confirms category inference source is valid)
    assert item_ids[0] == str(ARTICLE_ID), (
        f"Rank-1 must be self-retrieved article {ARTICLE_ID}, got {item_ids[0]}"
    )

    # Category coherence: inferred from rank-1's category ("Shirts")
    rank1_cat = art_map[ARTICLE_ID].get("category", "")
    result_cats = [
        art_map.get(int(iid), {}).get("category", "UNKNOWN")
        for iid in item_ids
        if iid.isdigit()
    ]
    off_cat = [c for c in result_cats if c != rank1_cat]
    assert not off_cat, (
        f"Category scatter in pure-image path (no item_id): expected all '{rank1_cat}', "
        f"got off-category items: {off_cat}. "
        f"The reranker must now infer category from rank-1 even without item_id."
    )

    # Price coherence: no 3.5x outliers vs rank-1 price
    rank1_price = float(art_map[ARTICLE_ID].get("price_inr") or 0)
    if rank1_price > 0:
        for iid in item_ids:
            if not iid.isdigit():
                continue
            item_price = float(art_map.get(int(iid), {}).get("price_inr") or 0)
            if item_price > 0:
                ratio = item_price / rank1_price
                assert ratio <= 3.5, (
                    f"Price outlier in results: item {iid} costs ₹{item_price:.0f} "
                    f"vs query ₹{rank1_price:.0f} (ratio={ratio:.1f}x). "
                    f"Price reranker should suppress this."
                )


def test_visual_search_inferred_category_mocked_fast() -> None:
    """Mocked: verify the route infers category from rank-1 and filters off-category items.

    FAISS returns: Shirt (rank-1, score=1.0), Overshirt (rank-2, score=0.93),
    then more Shirts.  With NO item_id, the route must infer 'Shirts' from rank-1
    and apply the reranker, pushing the Overshirt out of the top-k results.
    """
    import io
    from PIL import Image
    from app.rerank import RerankConfig

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(100, 150, 200)).save(buf, format="PNG")
    buf.seek(0)
    tiny_png = buf.read()

    SHIRT_A  = ARTICLE_ID       # rank-1: Shirt, ₹1299 — category inference source
    OVERSHIRT = ARTICLE_ID + 1  # rank-2: Overshirt — should be reranked down
    SHIRT_B  = ARTICLE_ID + 2   # rank-3: Shirt
    SHIRT_C  = ARTICLE_ID + 3   # rank-4: Shirt

    art_map = {
        SHIRT_A:   {"title": "Cotton Check Shirt",       "category": "Shirts",    "price_inr": 1299.0},
        OVERSHIRT: {"title": "Relaxed Fit Overshirt",    "category": "Overshirt", "price_inr": 1399.0},
        SHIRT_B:   {"title": "Linen Regular Fit Shirt",  "category": "Shirts",    "price_inr": 1199.0},
        SHIRT_C:   {"title": "Slim Fit Stretch Shirt",   "category": "Shirts",    "price_inr": 1399.0},
    }

    fixed_vec = np.zeros(512, dtype=np.float32)
    fixed_vec[0] = 1.0

    state = MagicMock()
    state.api_key = "vs-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.visual_retriever = MagicMock()
    state.visual_retriever.search.return_value = [
        (SHIRT_A,   1.00),
        (OVERSHIRT, 0.93),
        (SHIRT_B,   0.91),
        (SHIRT_C,   0.89),
    ]
    state.config.rerank = RerankConfig(
        enabled=True,
        candidate_pool_size=20,
        w_similarity=0.70,
        w_price_penalty=0.10,
        w_category_affinity=0.20,
        price_norm_inr=500.0,
        equivalent_group_bonus=0.70,
        w_diversity=0.0,
    )
    # C3: no min-score check fires in this mocked test.
    state.config.visual_search_min_score = None
    # C1: empty color index so color_rerank is a no-op here.
    state.color_index = {}

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == BRAND else None
    registry.brand_names.return_value = [BRAND]

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=fixed_vec),
    ):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            # NO item_id — pure image upload
            resp = client.post(
                f"/v1/{BRAND}/visual-search",
                files={"image": ("test.png", tiny_png, "image/png")},
                params={"k": 3},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    returned_ids = [r["item_id"] for r in resp.json()["results"]]

    assert returned_ids[0] == str(SHIRT_A), (
        f"Rank-1 must be the shirt that seeded category inference. Got: {returned_ids}"
    )
    assert str(OVERSHIRT) not in returned_ids, (
        f"Overshirt must be reranked out of top-3 after category inference. "
        f"Got: {returned_ids}. The inferred-category reranker is not filtering it."
    )


def test_visual_search_self_retrieval_mocked_fast() -> None:
    """Fast CI proxy: mocked FAISS always returns the seed item at rank 1.

    This test doesn't need CLIP or real indices — it verifies the route
    correctly propagates the visual_retriever result including rank-1.
    Run alongside the real-CLIP tests; this catches regressions in HTTP plumbing.
    """
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(100, 150, 200)).save(buf, format="PNG")
    buf.seek(0)
    tiny_png = buf.read()

    article_ids = [ARTICLE_ID, ARTICLE_ID + 1, ARTICLE_ID + 2]
    art_map = {
        aid: {
            "title": f"Item {aid}",
            "category": "Shirts",
            "price_inr": 999.0 + aid,
            "pdp_url": f"https://brand.com/{aid}",
        }
        for aid in article_ids
    }

    fixed_vec = np.zeros(512, dtype=np.float32)
    fixed_vec[0] = 1.0

    from unittest.mock import MagicMock, patch
    from app.rerank import RerankConfig

    state = MagicMock()
    state.api_key = "vs-test-key"
    state.config.brand = BRAND
    state.art_map = art_map
    state.visual_retriever = MagicMock()
    state.visual_retriever.search.return_value = [
        (aid, 1.0 - 0.01 * i) for i, aid in enumerate(article_ids)
    ]
    state.config.rerank = RerankConfig(enabled=False)
    # C3: no min-score check fires in this mocked test.
    state.config.visual_search_min_score = None
    # C1: empty color index so color_rerank is a no-op here.
    state.color_index = {}

    registry = MagicMock()
    registry.get.side_effect = lambda b: state if b == BRAND else None
    registry.brand_names.return_value = [BRAND]

    with (
        patch("app.api.main.load_registry", return_value=registry),
        patch("app.visual.encode_query_image", return_value=fixed_vec),
    ):
        from app.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                f"/v1/{BRAND}/visual-search",
                files={"image": ("test.png", tiny_png, "image/png")},
                params={"k": len(article_ids)},
                headers={"X-Api-Key": "vs-test-key"},
            )

    assert resp.status_code == 200, resp.text
    returned_ids = [r["item_id"] for r in resp.json()["results"]]
    assert returned_ids[0] == str(ARTICLE_ID), (
        f"Expected rank-1 item_id={ARTICLE_ID}, got {returned_ids}"
    )
