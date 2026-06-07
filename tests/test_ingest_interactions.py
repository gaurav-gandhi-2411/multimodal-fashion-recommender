from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.interactions import (
    build_product_mapping,
    load_generic_csv,
    load_shopify_orders_csv,
    process_interactions,
    split_chronological,
    write_splits,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _write_catalog(tmp_path: Path) -> Path:
    """Write a minimal catalog parquet with 5 items."""
    df = pd.DataFrame(
        {
            "article_id": [1, 2, 3, 4, 5],
            "product_id": ["SN001", "SN002", "SN003", "SN004", "SN005"],
            "title": ["T1", "T2", "T3", "T4", "T5"],
        }
    )
    p = tmp_path / "data" / "snitch" / "items.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def _generic_csv(tmp_path: Path) -> Path:
    """Write a generic events CSV with 10 rows across 2 users."""
    content = textwrap.dedent(
        """\
        user_id,product_id,timestamp,event_type
        u1,SN001,2025-01-01T10:00:00Z,purchase
        u1,SN002,2025-01-02T11:00:00Z,view
        u1,SN003,2025-01-03T12:00:00Z,wishlist
        u2,SN001,2025-01-04T09:00:00Z,purchase
        u2,SN004,2025-01-05T10:00:00Z,cart
        u1,SN002,2025-01-06T14:00:00Z,purchase
        u2,SN003,2025-01-07T15:00:00Z,view
        u1,SN005,2025-01-08T16:00:00Z,purchase
        u2,SN005,2025-01-09T17:00:00Z,wishlist
        u1,SN004,2025-01-10T18:00:00Z,view
        """
    )
    p = tmp_path / "events.csv"
    p.write_text(content)
    return p


def _shopify_csv(tmp_path: Path) -> Path:
    """Write a Shopify orders export CSV."""
    content = textwrap.dedent(
        """\
        Email,Paid at,Lineitem sku,Lineitem name
        alice@example.com,2025-01-01T10:00:00+00:00,SN001,Oversized Tee
        alice@example.com,2025-01-03T12:00:00+00:00,SN003,Cargo Pants
        bob@example.com,2025-01-05T09:00:00+00:00,SN002,Slim Fit Shirt
        bob@example.com,2025-01-07T11:00:00+00:00,SN004,Bomber Jacket
        """
    )
    p = tmp_path / "shopify_orders.csv"
    p.write_text(content)
    return p


# ── TestLoadGenericCsv ─────────────────────────────────────────────────────────


class TestLoadGenericCsv:
    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        df = load_generic_csv(_generic_csv(tmp_path))
        assert len(df) == 10
        assert set(df.columns) == {"user_id", "product_id", "timestamp", "event_type"}

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("user_id,product_id,timestamp\nu1,SN001,2025-01-01T00:00:00Z\n")
        with pytest.raises(ValueError, match="event_type"):
            load_generic_csv(p)

    def test_rows_with_null_fields_dropped(self, tmp_path: Path) -> None:
        p = tmp_path / "null_rows.csv"
        p.write_text(
            "user_id,product_id,timestamp,event_type\n"
            "u1,SN001,2025-01-01T00:00:00Z,purchase\n"
            ",SN002,2025-01-02T00:00:00Z,view\n"
        )
        df = load_generic_csv(p)
        assert len(df) == 1


# ── TestLoadShopifyCsv ─────────────────────────────────────────────────────────


class TestLoadShopifyCsv:
    def test_loads_valid_shopify_csv(self, tmp_path: Path) -> None:
        df = load_shopify_orders_csv(_shopify_csv(tmp_path))
        assert len(df) == 4
        assert set(df.columns) == {"user_id", "product_id", "timestamp", "event_type"}

    def test_event_type_is_purchase(self, tmp_path: Path) -> None:
        df = load_shopify_orders_csv(_shopify_csv(tmp_path))
        assert (df["event_type"] == "purchase").all()

    def test_email_normalised_lowercase(self, tmp_path: Path) -> None:
        p = tmp_path / "caps.csv"
        p.write_text(
            "Email,Paid at,Lineitem sku\n"
            "Alice@Example.COM,2025-01-01T00:00:00+00:00,SN001\n"
        )
        df = load_shopify_orders_csv(p)
        assert df["user_id"].iloc[0] == "alice@example.com"

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("Email,Paid at\nalice@x.com,2025-01-01T00:00:00+00:00\n")
        with pytest.raises(ValueError, match="Lineitem sku"):
            load_shopify_orders_csv(p)


# ── TestBuildProductMapping ────────────────────────────────────────────────────


class TestBuildProductMapping:
    def test_returns_correct_mapping(self, tmp_path: Path) -> None:
        catalog = _write_catalog(tmp_path)
        mapping = build_product_mapping(catalog)
        assert mapping["SN001"] == 1
        assert mapping["SN005"] == 5

    def test_missing_catalog_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ingest_catalog"):
            build_product_mapping(tmp_path / "nonexistent" / "items.parquet")


# ── TestProcessInteractions ───────────────────────────────────────────────────


class TestProcessInteractions:
    def test_maps_product_id_to_article_id(self, tmp_path: Path) -> None:
        catalog = _write_catalog(tmp_path)
        mapping = build_product_mapping(catalog)
        raw = load_generic_csv(_generic_csv(tmp_path))
        df = process_interactions(raw, mapping)
        assert "article_id" in df.columns
        assert df["article_id"].dtype == int

    def test_unknown_product_ids_dropped(self, tmp_path: Path) -> None:
        catalog = _write_catalog(tmp_path)
        mapping = build_product_mapping(catalog)
        p = tmp_path / "with_unknown.csv"
        p.write_text(
            "user_id,product_id,timestamp,event_type\n"
            "u1,SN001,2025-01-01T00:00:00Z,purchase\n"
            "u1,UNKNOWN99,2025-01-02T00:00:00Z,view\n"
        )
        raw = load_generic_csv(p)
        df = process_interactions(raw, mapping)
        assert len(df) == 1
        assert df["article_id"].iloc[0] == 1

    def test_output_columns_are_correct(self, tmp_path: Path) -> None:
        catalog = _write_catalog(tmp_path)
        mapping = build_product_mapping(catalog)
        raw = load_generic_csv(_generic_csv(tmp_path))
        df = process_interactions(raw, mapping)
        assert set(df.columns) == {"customer_id", "article_id", "t_dat"}

    def test_sorted_by_date(self, tmp_path: Path) -> None:
        catalog = _write_catalog(tmp_path)
        mapping = build_product_mapping(catalog)
        raw = load_generic_csv(_generic_csv(tmp_path))
        df = process_interactions(raw, mapping)
        dates = list(df["t_dat"])
        assert dates == sorted(dates)


# ── TestSplitChronological ────────────────────────────────────────────────────


class TestSplitChronological:
    def _make_df(self, n: int) -> pd.DataFrame:
        import datetime
        return pd.DataFrame(
            {
                "customer_id": [f"u{i}" for i in range(n)],
                "article_id": list(range(1, n + 1)),
                "t_dat": [
                    datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(n)
                ],
            }
        )

    def test_split_sizes_sum_to_total(self) -> None:
        df = self._make_df(100)
        train, val, test = split_chronological(df)
        assert len(train) + len(val) + len(test) == 100

    def test_default_split_is_80_10_10(self) -> None:
        df = self._make_df(100)
        train, val, test = split_chronological(df)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_chronological_order_preserved(self) -> None:
        df = self._make_df(100)
        train, val, test = split_chronological(df)
        # train max date < val min date < test min date
        assert max(train["t_dat"]) <= min(val["t_dat"])
        assert max(val["t_dat"]) <= min(test["t_dat"])


# ── TestWriteSplits ───────────────────────────────────────────────────────────


class TestWriteSplits:
    def test_creates_three_parquets(self, tmp_path: Path) -> None:
        import datetime
        df = pd.DataFrame(
            {
                "customer_id": ["u1", "u2", "u3"],
                "article_id": [1, 2, 3],
                "t_dat": [datetime.date(2025, 1, i) for i in range(1, 4)],
            }
        )
        train, val, test = split_chronological(df)
        out = tmp_path / "transactions"
        write_splits((train, val, test), out)
        assert (out / "train.parquet").exists()
        assert (out / "val.parquet").exists()
        assert (out / "test.parquet").exists()

    def test_parquet_columns_correct(self, tmp_path: Path) -> None:
        import datetime
        df = pd.DataFrame(
            {
                "customer_id": [f"u{i}" for i in range(20)],
                "article_id": list(range(1, 21)),
                "t_dat": [
                    datetime.date(2025, 1, 1) + datetime.timedelta(days=i)
                    for i in range(20)
                ],
            }
        )
        out = tmp_path / "transactions"
        write_splits(split_chronological(df), out)
        train_df = pd.read_parquet(out / "train.parquet")
        assert set(train_df.columns) == {"customer_id", "article_id", "t_dat"}
