"""
CLI: ingest interaction data for a brand and produce train/val/test parquets.

Usage
-----
# Generic events CSV (columns: user_id, product_id, timestamp, event_type)
python scripts/ingest_interactions.py --brand snitch --path orders.csv

# Shopify orders export
python scripts/ingest_interactions.py --brand snitch --path shopify_orders.csv --format shopify

After running this, restart the API server — personalized /recommend unlocks for
any customer_id that appears in the ingested data.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.interactions import (  # noqa: E402
    build_product_mapping,
    load_generic_csv,
    load_shopify_orders_csv,
    process_interactions,
    split_chronological,
    write_splits,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest interaction data for a brand.",
    )
    p.add_argument("--brand", required=True, help="Brand slug (e.g. snitch)")
    p.add_argument("--path", required=True, help="Path to the interactions CSV")
    p.add_argument(
        "--format",
        choices=["generic", "shopify"],
        default="generic",
        help="CSV format: 'generic' (user_id/product_id/timestamp/event_type) "
        "or 'shopify' (Email/Paid at/Lineitem sku). Default: generic",
    )
    p.add_argument(
        "--output-base",
        default=".",
        help="Root output directory (default: repo root)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_base = Path(args.output_base)
    csv_path = Path(args.path)
    brand = args.brand

    catalog_parquet = output_base / "data" / brand / "items.parquet"
    transactions_dir = output_base / "data" / brand / "transactions"
    brand_yaml = output_base / "brands" / f"{brand}.yaml"

    # 1. Load catalog mapping
    try:
        product_map = build_product_mapping(catalog_parquet)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Loaded catalog: %d products", len(product_map))

    # 2. Load interactions
    try:
        if args.format == "shopify":
            raw_df = load_shopify_orders_csv(csv_path)
        else:
            raw_df = load_generic_csv(csv_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Failed to load CSV: %s", exc)
        return 1
    logger.info("Loaded %d raw interaction rows", len(raw_df))

    # 3. Map + filter
    mapped_df = process_interactions(raw_df, product_map)
    if mapped_df.empty:
        logger.error("No interactions remain after mapping — check product_ids match catalog")
        return 1
    logger.info("%d rows after mapping to article_ids", len(mapped_df))

    # 4. Chronological split
    train, val, test = split_chronological(mapped_df)
    logger.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))

    # 5. Write parquets
    write_splits((train, val, test), transactions_dir)

    # 6. Update brand YAML
    if brand_yaml.exists():
        with open(brand_yaml) as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg["transactions_dir"] = f"data/{brand}/transactions"
        brand_yaml.write_text(
            yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )
        logger.info("Updated %s with transactions_dir", brand_yaml)
    else:
        logger.warning(
            "brands/%s.yaml not found — run ingest_catalog.py first to create it. "
            "Parquets were written but the YAML was not updated.",
            brand,
        )

    print(f"\nInteraction data ingested for brand '{brand}'.")
    print(f"  Train: {len(train)} rows")
    print(f"  Val:   {len(val)} rows")
    print(f"  Test:  {len(test)} rows")
    print("  Personalized /recommend is now available for ingested customer IDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
