"""Smoke tests for Phase 3 Indian brand demo."""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent

BRANDS = ["snitch", "fashor", "powerlook"]

# Minimum row counts validated at ingest time
MIN_ROWS: dict[str, int] = {"snitch": 490, "fashor": 3500, "powerlook": 890}

# Expected columns in each artifact
CATALOG_CSV_COLS = {
    "product_id", "title", "description", "image_url", "price_inr", "category", "pdp_url"
}
ITEMS_PARQUET_COLS = {
    "article_id", "product_id", "title", "category", "image_url", "pdp_url", "price_inr"
}
USERS_CSV_COLS = {"user_id", "product_id", "timestamp", "event_type"}

FAISS_DIM = 256


# ---------------------------------------------------------------------------
# Group 1: Catalog data integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("brand", BRANDS)
def test_catalog_csv_columns(brand: str) -> None:
    """catalog.csv exists, has the right 7 columns, no nulls, sane price/URL values."""
    csv_path = ROOT / "data" / brand / "catalog.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)

    assert set(df.columns) == CATALOG_CSV_COLS, (
        f"{brand}: unexpected columns {set(df.columns)}"
    )
    assert df.isnull().sum().sum() == 0, f"{brand}: catalog.csv has null values"
    assert (df["price_inr"] > 0).all(), f"{brand}: price_inr must be > 0 for all rows"
    assert df["pdp_url"].str.startswith("https://").all(), (
        f"{brand}: all pdp_url values must start with 'https://'"
    )


@pytest.mark.parametrize("brand", BRANDS)
def test_items_parquet_columns(brand: str) -> None:
    """items.parquet exists, has correct columns, int article_id, and meets row-count minimums."""
    parquet_path = ROOT / "data" / brand / "items.parquet"
    assert parquet_path.exists(), f"Missing {parquet_path}"

    df = pd.read_parquet(parquet_path)

    assert ITEMS_PARQUET_COLS.issubset(set(df.columns)), (
        f"{brand}: missing columns {ITEMS_PARQUET_COLS - set(df.columns)}"
    )
    assert pd.api.types.is_integer_dtype(df["article_id"]), (
        f"{brand}: article_id must be integer dtype, got {df['article_id'].dtype}"
    )
    assert len(df) >= MIN_ROWS[brand], (
        f"{brand}: expected ≥ {MIN_ROWS[brand]} rows, got {len(df)}"
    )


# ---------------------------------------------------------------------------
# Group 2: Brand config files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("brand", BRANDS)
def test_brand_yaml_structure(brand: str) -> None:
    """brands/{brand}.yaml exists and has all required keys with correct value patterns."""
    yaml_path = ROOT / "brands" / f"{brand}.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with yaml_path.open() as fh:
        cfg = yaml.safe_load(fh)

    required_keys = {"brand", "display_name", "catalog_path", "index_path", "api_key_env"}
    missing = required_keys - set(cfg.keys())
    assert not missing, f"{brand}: YAML missing keys {missing}"

    assert cfg["catalog_path"], f"{brand}: catalog_path must be a non-empty string"
    assert cfg["api_key_env"] == f"{brand.upper()}_API_KEY", (
        f"{brand}: api_key_env expected '{brand.upper()}_API_KEY', got '{cfg['api_key_env']}'"
    )


# ---------------------------------------------------------------------------
# Group 3: Indian vocabulary in Fashor catalog
# ---------------------------------------------------------------------------


def test_fashor_has_ethnic_vocabulary() -> None:
    """At least 50 % of Fashor rows have an ethnic-wear category keyword."""
    df = pd.read_parquet(ROOT / "data" / "fashor" / "items.parquet")
    ethnic_terms = ["kurta", "Kurta", "2P", "3P"]
    pattern = "|".join(ethnic_terms)
    ethnic_mask = df["category"].str.contains(pattern, na=False)
    fraction = ethnic_mask.mean()
    assert fraction >= 0.50, (
        f"Expected ≥ 50 % Fashor rows with ethnic vocabulary, got {fraction:.1%}"
    )


# ---------------------------------------------------------------------------
# Group 4: Synthetic users labelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("brand", BRANDS)
def test_synthetic_users_labelled(brand: str) -> None:
    """synthetic_users.csv exists and all rows are correctly labelled purchase events."""
    csv_path = ROOT / "data" / brand / "synthetic_users.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)

    assert USERS_CSV_COLS.issubset(set(df.columns)), (
        f"{brand}: missing columns {USERS_CSV_COLS - set(df.columns)}"
    )
    assert df["user_id"].str.startswith("synthetic_").all(), (
        f"{brand}: not all user_id values start with 'synthetic_'"
    )
    assert (df["event_type"] == "purchase").all(), (
        f"{brand}: expected all event_type == 'purchase'"
    )


# ---------------------------------------------------------------------------
# Group 5: FAISS index loads and searches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("brand", BRANDS)
def test_faiss_index_searchable(brand: str) -> None:
    """FAISS index loads, is non-empty, and handles a zero-vector query without error."""
    import faiss  # noqa: PLC0415

    index_path = ROOT / "indices" / brand / "active.faiss" / "faiss.index"
    assert index_path.exists(), f"Missing {index_path}"

    index = faiss.read_index(str(index_path))
    assert index.ntotal > 0, f"{brand}: FAISS index is empty"

    query = np.zeros((1, FAISS_DIM), dtype=np.float32)
    k = 5
    distances, ids = index.search(query, k)
    assert distances.shape == (1, k), f"{brand}: unexpected distances shape {distances.shape}"
    assert ids.shape == (1, k), f"{brand}: unexpected ids shape {ids.shape}"


# ---------------------------------------------------------------------------
# Group 6: API smoke tests — /similar and /recommend
# ---------------------------------------------------------------------------


def _make_brand_state(brand: str) -> MagicMock:
    """Build a minimal BrandState mock for a given Indian brand using real article IDs."""
    df = pd.read_parquet(ROOT / "data" / brand / "items.parquet")

    # Load real FAISS index so reconstruct() works correctly
    import faiss  # noqa: PLC0415

    index_path = str(ROOT / "indices" / brand / "active.faiss" / "faiss.index")
    real_index = faiss.read_index(index_path)

    aids_path = ROOT / "indices" / brand / "active.faiss" / "article_ids.pkl"
    with aids_path.open("rb") as fh:
        raw_aids = pickle.load(fh)  # noqa: S301

    # article_ids in the pkl are strings "1", "2", … — normalise to int
    article_ids = [int(a) for a in raw_aids]
    faiss_aid_to_row = {aid: i for i, aid in enumerate(article_ids)}

    art_map: dict[int, dict] = df.set_index("article_id").to_dict("index")

    # Wrap the real FAISS index in a retriever-like mock so search() returns typed pairs
    retriever = MagicMock()
    retriever.index = real_index
    retriever.article_ids = article_ids

    def _search(query_emb: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        dists, ids = real_index.search(q, k)
        return [(int(ids[0, i]), float(dists[0, i])) for i in range(k) if ids[0, i] != -1]

    retriever.search.side_effect = _search

    cfg_data = yaml.safe_load((ROOT / "brands" / f"{brand}.yaml").read_text())

    state = MagicMock()
    state.config.brand = brand
    state.config.display_name = cfg_data["display_name"]
    state.config.llm.enabled = False
    state.config.llm.provider = "template"
    state.api_key = "test-key"
    state.retriever = retriever
    state.art_map = art_map
    state.faiss_aid_to_row = faiss_aid_to_row
    state.user_history = None  # no transaction history loaded in smoke tests
    return state


@pytest.fixture(scope="module")
def indian_registry() -> MagicMock:
    """A BrandRegistry mock covering all three Indian brands."""
    states = {brand: _make_brand_state(brand) for brand in BRANDS}

    registry = MagicMock()
    registry.get.side_effect = lambda brand: states.get(brand)
    registry.brand_names.return_value = list(states.keys())
    return registry


@pytest.fixture(scope="module")
def indian_client(indian_registry: MagicMock):
    """FastAPI TestClient wired to the Indian brand registry mock."""
    from fastapi.testclient import TestClient

    with patch("app.api.main.load_registry", return_value=indian_registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.mark.parametrize("brand", BRANDS)
def test_similar_returns_results(brand: str, indian_client) -> None:
    """/similar returns ≥ 1 result with item_id and score for each Indian brand."""
    df = pd.read_parquet(ROOT / "data" / brand / "items.parquet")
    first_item_id = str(df["article_id"].iloc[0])

    resp = indian_client.get(
        f"/v1/{brand}/item/{first_item_id}/similar?k=3",
        headers={"X-Api-Key": "test-key"},
    )
    assert resp.status_code == 200, f"{brand}: {resp.status_code} — {resp.text}"
    data = resp.json()
    assert "results" in data, f"{brand}: response missing 'results' key"
    assert len(data["results"]) >= 1, f"{brand}: expected ≥ 1 result, got {len(data['results'])}"
    for item in data["results"]:
        assert "item_id" in item, f"{brand}: result missing 'item_id'"
        assert "score" in item, f"{brand}: result missing 'score'"


@pytest.mark.parametrize("brand", BRANDS)
def test_recommend_synthetic_user(brand: str, indian_client) -> None:
    """/recommend returns 200 for a synthetic user (cold_start=True is acceptable)."""
    users_df = pd.read_csv(ROOT / "data" / brand / "synthetic_users.csv")
    user_id = users_df["user_id"].iloc[0]

    # Provide item_id as cold-start fallback: when user_history=None the API needs
    # item_id to produce an embedding and return 200 instead of 422.
    items_df = pd.read_parquet(ROOT / "data" / brand / "items.parquet")
    item_id = str(items_df["article_id"].iloc[0])

    resp = indian_client.post(
        f"/v1/{brand}/recommend",
        json={"user_id": user_id, "item_id": item_id, "k": 3},
        headers={"X-Api-Key": "test-key"},
    )
    assert resp.status_code == 200, f"{brand}: {resp.status_code} — {resp.text}"
    data = resp.json()
    assert "results" in data, f"{brand}: response missing 'results' key"
    assert isinstance(data["results"], list), f"{brand}: 'results' must be a list"
