"""
Ingest a brand catalog from a Shopify /products.json endpoint or a CSV file.

Usage — CSV source:
    python scripts/ingest_catalog.py --source csv --path catalog.csv --brand snitch

Usage — Shopify source:
    python scripts/ingest_catalog.py --source shopify --url https://snitch.co.in --brand snitch

Outputs (relative to project root):
    brands/<brand>.yaml            ready for the Phase 1 API (no restart needed)
    data/<brand>/items.parquet     catalog with article_id (int) + original product_id
    indices/<brand>/active.faiss/  FAISS IndexFlatIP directory
    indices/<brand>/item_emb.npy   256-dim fused embeddings

Cold-start is the baseline: the brand is immediately queryable via
    GET /v1/<brand>/item/{article_id}/similar
with no interaction data. Interactions are an optional second step.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import run_catalog_pipeline
from app.ingestion.sources import CsvSource, ShopifySource


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a brand catalog into the fashion recommender index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["csv", "shopify"],
        required=True,
        help="Data source: 'csv' for a local CSV file, 'shopify' for a Shopify store URL.",
    )
    parser.add_argument(
        "--brand",
        required=True,
        help="Brand slug used in URL paths (/v1/{brand}/...) and output file names.",
    )
    parser.add_argument(
        "--path",
        help="Path to catalog CSV file. Required when --source csv.",
    )
    parser.add_argument(
        "--url",
        help=(
            "Shopify store URL or direct /products.json URL. "
            "Required when --source shopify."
        ),
    )
    parser.add_argument(
        "--output-base",
        default=".",
        help="Base directory for all output files (default: current directory).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Path to TwoTowerModel checkpoint (default: checkpoints/best.pt).",
    )
    parser.add_argument(
        "--respect-robots",
        action="store_true",
        default=False,
        help=(
            "Enforce robots.txt for the Shopify source. "
            "Default: warn and proceed (safe for authorized client onboarding)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent image download workers (default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Validate source-specific required arguments
    if args.source == "csv":
        if not args.path:
            print("ERROR: --path is required when --source csv", file=sys.stderr)
            return 1
        source = CsvSource(Path(args.path))
    else:
        if not args.url:
            print("ERROR: --url is required when --source shopify", file=sys.stderr)
            return 1
        source = ShopifySource(args.url, respect_robots=args.respect_robots)

    print(f"Fetching catalog from {args.source} source...")
    try:
        items = source.fetch()
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR fetching catalog: {exc}", file=sys.stderr)
        return 1

    if not items:
        print("ERROR: no valid items found in catalog.", file=sys.stderr)
        return 1

    print(f"  {len(items)} valid items fetched.")
    print(f"\nRunning ingestion pipeline for brand '{args.brand}'...")

    try:
        yaml_path = run_catalog_pipeline(
            items,
            args.brand,
            output_base=Path(args.output_base),
            checkpoint_path=Path(args.checkpoint),
            max_download_workers=args.workers,
        )
    except Exception as exc:
        print(f"ERROR during ingestion pipeline: {exc}", file=sys.stderr)
        return 1

    brand_upper = args.brand.upper()
    print("\n✓ Ingestion complete.")
    print(f"  Brand YAML : {yaml_path}")
    print("\nNext steps:")
    print(f"  1. Set the API key:  export {brand_upper}_API_KEY=<your-key>")
    print("  2. Start the server: uvicorn app.api.main:app --reload")
    print("  3. Test cold-start:  curl -H 'X-Api-Key: <key>' \\")
    print(f"                            http://localhost:8000/v1/{args.brand}/item/1/similar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
