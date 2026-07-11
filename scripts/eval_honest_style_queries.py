"""scripts/eval_honest_style_queries.py

Honest free-text style-search eval: measures category recall@k using human-written,
realistic style queries ("something for a summer wedding") instead of the item's own
product title.

Why this exists
----------------
The locked 92.5% style-search pitch number (`scripts/eval_serve_path.py`) queries the API
with the ITEM'S OWN TITLE as the search text (e.g. "Solid Cotton Kurta"). That's close to
lexical retrieval, not a realistic free-text style query, and likely overstates how the
system performs against how a real user (or a buyer's storefront search box) would
actually query it. This script measures the SAME metric (category recall@k) through the
SAME serve path (`run_local_style`/`run_http_style` from `eval_serve_path.py` -- reused,
not reimplemented, so this can never silently diverge from the real serve path) but with
30 human-written queries per brand (`evals/fixtures/style_search_queries/{brand}.json`),
each independent of any item's actual title.

This number is reported ALONGSIDE the title-derived 92.5%, not as a replacement for it --
both are real, they measure different things, and conflating them would be dishonest.

Usage:
    python scripts/eval_honest_style_queries.py --brand snitch --local
    python scripts/eval_honest_style_queries.py --brand snitch --local --http-mode \\
        --api-base https://fashion-recommender-staging-rm7rz66wza-el.a.run.app \\
        --api-key <key>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FIXTURES_DIR = REPO_ROOT / "evals" / "fixtures" / "style_search_queries"


def load_honest_queries(brand: str) -> list[SimpleNamespace]:
    """Load `evals/fixtures/style_search_queries/{brand}.json` as fake catalog-row objects.

    `run_local_style`/`run_http_style` (from `eval_serve_path.py`) read `.title` and
    `.category` off each row via `getattr` (duck-typed against `DataFrame.itertuples()`
    rows) -- a `SimpleNamespace` with those two attributes satisfies that contract exactly,
    so this reuses the real serve-path evaluation functions unmodified.
    """
    path = FIXTURES_DIR / f"{brand}.json"
    if not path.exists():
        raise FileNotFoundError(f"No honest query fixture for brand {brand!r}: {path}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return [SimpleNamespace(title=r["query"], category=r["expected_category"]) for r in rows]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--brand", default="snitch")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--http-mode", action="store_true", dest="http_mode")
    parser.add_argument(
        "--api-base",
        default="https://fashion-recommender-staging-rm7rz66wza-el.a.run.app",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--delay", type=float, default=1.1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.local and not args.http_mode:
        args.local = True

    import pandas as pd
    import yaml
    from eval_serve_path import run_http_style, run_local_style  # noqa: E402

    from app.brands.registry import BrandConfig
    from src.retrieval.faiss_index import FaissRetriever

    brand_yaml_path = REPO_ROOT / "brands" / f"{args.brand}.yaml"
    with brand_yaml_path.open() as f:
        brand_cfg = BrandConfig.model_validate(yaml.safe_load(f))
    rerank_cfg = brand_cfg.rerank

    catalog = pd.read_parquet(REPO_ROOT / "data" / args.brand / "items.parquet")
    catalog["article_id"] = catalog["article_id"].astype(int)

    honest_rows = load_honest_queries(args.brand)
    print(f"Brand: {args.brand}  |  {len(honest_rows)} human-written queries  |  k={args.k}")
    print("These queries are NOT derived from any item's title -- independent, realistic")
    print("free-text style queries, each mapped to a plausible target category.\n")

    if args.local:
        retriever = FaissRetriever.load(brand_cfg.visual_index_path)
        index_ids = {int(x) for x in retriever.article_ids}
        catalog_indexed = catalog[catalog["article_id"].isin(index_ids)].reset_index(drop=True)
        art_map = catalog_indexed.set_index("article_id").to_dict("index")

        local_result = run_local_style(
            catalog_indexed, retriever, args.k, honest_rows, art_map, rerank_cfg
        )
        print(
            f"LOCAL  honest-query category recall@{args.k}: "
            f"{local_result['recall']:.4f}  ({local_result['hits']}/{local_result['total']})"
        )

    if args.http_mode:
        if not args.api_key:
            print("ERROR: --api-key required with --http-mode", file=sys.stderr)
            return 1
        aid_to_cat = {int(r.article_id): str(r.category or "") for r in catalog.itertuples()}
        http_result = run_http_style(
            honest_rows, aid_to_cat, args.api_base, args.api_key, args.brand, args.k, args.delay
        )
        print(
            f"HTTP   honest-query category recall@{args.k}: "
            f"{http_result['recall']:.4f}  ({http_result['hits']}/{http_result['total']})"
        )

    print(
        "\nCompare against the title-derived pitch number for the same brand/k "
        "(scripts/eval_serve_path.py --mode style) -- report both, labeled, never conflated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
