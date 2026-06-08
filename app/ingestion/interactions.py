from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── loaders ───────────────────────────────────────────────────────────────────


def load_generic_csv(path: Path) -> pd.DataFrame:
    """
    Load a generic events CSV with required columns:
        user_id, product_id, timestamp, event_type
    Returns a DataFrame with those four columns; invalid rows are dropped
    with a warning. Raises ValueError if required columns are missing.
    """
    REQUIRED = {"user_id", "product_id", "timestamp", "event_type"}
    # product_id must be str so it matches the string keys in the catalog parquet.
    # Without this, pandas infers large numeric-looking IDs (e.g. "8713437249698")
    # as int64, causing 100% row drop when mapped against string catalog product_ids.
    df = pd.read_csv(path, dtype={"product_id": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Generic CSV missing required columns: {sorted(missing)}")
    before = len(df)
    df = df.dropna(subset=list(REQUIRED))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if len(df) < before:
        logger.warning("Dropped %d rows with missing/invalid fields", before - len(df))
    return df[list(REQUIRED)].reset_index(drop=True)


def load_shopify_orders_csv(path: Path) -> pd.DataFrame:
    """
    Load a Shopify orders export CSV. Expected columns:
        Email, Paid at, Lineitem sku
    Returns a DataFrame with columns user_id, product_id, timestamp, event_type.
    Rows with missing Email, Paid at, or Lineitem sku are dropped with a warning.
    Raises ValueError if the three required Shopify columns are absent.
    """
    REQUIRED = {"Email", "Paid at", "Lineitem sku"}
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Shopify orders CSV missing required columns: {sorted(missing)}")
    before = len(df)
    df = df.dropna(subset=["Email", "Paid at", "Lineitem sku"])
    df["Paid at"] = pd.to_datetime(df["Paid at"], utc=True, errors="coerce")
    df = df.dropna(subset=["Paid at"])
    if len(df) < before:
        logger.warning("Dropped %d Shopify rows with missing/invalid fields", before - len(df))
    out = pd.DataFrame(
        {
            "user_id": df["Email"].str.strip().str.lower(),
            "product_id": df["Lineitem sku"].astype(str).str.strip(),
            "timestamp": df["Paid at"],
            "event_type": "purchase",
        }
    )
    return out.reset_index(drop=True)


# ── mapping + filtering ───────────────────────────────────────────────────────


def build_product_mapping(catalog_parquet: Path) -> dict[str, int]:
    """
    Read the catalog parquet and return {product_id: article_id}.
    Raises FileNotFoundError (with actionable message) if the parquet is absent.
    """
    if not catalog_parquet.exists():
        raise FileNotFoundError(
            f"Catalog parquet not found at {catalog_parquet}. "
            "Run `python scripts/ingest_catalog.py` for this brand first."
        )
    df = pd.read_parquet(catalog_parquet, columns=["product_id", "article_id"])
    return dict(zip(df["product_id"], df["article_id"].astype(int), strict=False))


def process_interactions(
    interactions_df: pd.DataFrame,
    product_map: dict[str, int],
) -> pd.DataFrame:
    """
    Map product_id → article_id, drop rows with unknown product_ids.
    Returns a DataFrame with columns: customer_id (str), article_id (int), t_dat (date).
    """
    df = interactions_df.copy()
    before = len(df)
    df["article_id"] = df["product_id"].map(product_map)
    n_unknown = df["article_id"].isna().sum()
    if n_unknown:
        logger.warning(
            "%d/%d interaction rows dropped — product_id not in catalog", n_unknown, before
        )
    df = df.dropna(subset=["article_id"])
    df["article_id"] = df["article_id"].astype(int)
    df["t_dat"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    return (
        df.rename(columns={"user_id": "customer_id"})[["customer_id", "article_id", "t_dat"]]
        .sort_values("t_dat")
        .reset_index(drop=True)
    )


# ── chronological split ───────────────────────────────────────────────────────


def split_chronological(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    80 / 10 / 10 chronological split.
    df must be sorted by t_dat (process_interactions guarantees this).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


# ── writer ────────────────────────────────────────────────────────────────────


def write_splits(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write train/val/test parquets to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")
    for name, split_df in zip(names, splits, strict=False):
        path = output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Wrote %d rows → %s", len(split_df), path)
