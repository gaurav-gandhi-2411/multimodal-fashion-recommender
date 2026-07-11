"""
Pre-flight check for a single brand's onboarding: validate its YAML config and confirm
every asset file it requires exists locally.

Usage:
    python scripts/brand_preflight.py --brand powerlook
    python scripts/brand_preflight.py --brand snitch --brands-dir brands --output-base .

Prints a single JSON object to stdout (machine-parseable — human diagnostics go to
stderr). Exit code 0 when every required path is present locally, 1 otherwise (the JSON
is still printed on failure so callers can report the missing list, not just a bare
failure code).

This CLI derives its required-paths list from `app.storage.brand_asset_paths`, the same
function the runtime GCS sync (`app.storage.sync_brand_assets`) uses — so this pre-flight
check and the container's startup sync can never diverge (see docs/architecture/adr/
0001-brand-onboarding-runbook.md for the incident history this closes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from pydantic import ValidationError

from app.brands.registry import BrandConfig
from app.storage import brand_asset_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-flight check: validate a brand YAML and confirm its required "
        "asset files exist locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--brand",
        required=True,
        help="Brand slug. Reads <brands-dir>/<brand>.yaml.",
    )
    parser.add_argument(
        "--brands-dir",
        default="brands",
        help="Directory containing brand YAML files (default: brands).",
    )
    parser.add_argument(
        "--output-base",
        default=".",
        help="Base directory that required paths are resolved relative to (default: "
        "current directory). Matches app/storage.py's convention: paths in brand YAML "
        "are repo-relative.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate the brand YAML, derive required paths, and report local presence as JSON."""
    args = _parse_args()

    yaml_path = Path(args.brands_dir) / f"{args.brand}.yaml"
    if not yaml_path.exists():
        print(f"ERROR: brand YAML not found: {yaml_path}", file=sys.stderr)
        return 1

    try:
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        cfg = BrandConfig.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        print(f"ERROR: invalid brand YAML {yaml_path}: {exc}", file=sys.stderr)
        return 1

    required_paths = brand_asset_paths(cfg)
    output_base = Path(args.output_base)
    missing_local = [p for p in required_paths if not (output_base / p).exists()]
    all_present = not missing_local

    result = {
        "brand": cfg.brand,
        "required_paths": required_paths,
        "missing_local": missing_local,
        "all_present": all_present,
    }
    print(json.dumps(result))

    if not all_present:
        print(
            f"MISSING {len(missing_local)}/{len(required_paths)} required local assets "
            f"for brand {cfg.brand!r}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"All {len(required_paths)} required local assets present for brand {cfg.brand!r}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
