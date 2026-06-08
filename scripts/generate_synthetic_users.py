from __future__ import annotations

"""Generate synthetic user interaction CSVs for three Indian fashion brands.

Outputs one CSV per brand with columns: user_id, product_id, timestamp, event_type.
All product_ids are verified to exist in the brand's catalog.csv.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2024-06-30")
EVENT_TYPE = "purchase"

# ── Brand × archetype configuration ──────────────────────────────────────────
# Each archetype entry:  (archetype_slug, n_users, category_weights)
# category_weights: dict[category_name -> float] — unnormalised; missing == 0


@dataclass
class ArchetypeConfig:
    slug: str
    n_users: int
    category_weights: dict[str, float]
    interactions_range: tuple[int, int] = field(default=(8, 15))


BRAND_CONFIG: dict[str, list[ArchetypeConfig]] = {
    "snitch": [
        ArchetypeConfig(
            slug="streetwear",
            n_users=5,
            category_weights={"Shirts": 3.0, "T-Shirts": 3.0, "Cargo Pants": 4.0},
        ),
        ArchetypeConfig(
            slug="minimalist",
            n_users=5,
            category_weights={"Shirts": 5.0, "Jeans": 5.0},
        ),
        ArchetypeConfig(
            slug="activecasual",
            n_users=5,
            category_weights={"T-Shirts": 4.0, "Sweaters": 2.0, "Trousers": 2.0, "Shorts": 2.0},
        ),
    ],
    "fashor": [
        ArchetypeConfig(
            slug="festive",
            n_users=5,
            category_weights={"3P Kurta Set": 5.0, "Kurta Set": 5.0},
        ),
        ArchetypeConfig(
            slug="officekurta",
            n_users=5,
            category_weights={"Kurtas": 4.0, "Kurta": 3.0, "2P Kurta Set": 3.0},
        ),
        ArchetypeConfig(
            slug="fusion",
            n_users=5,
            category_weights={"Dresses": 4.0, "Dress": 2.0, "Kurtas": 3.0, "Co-ord Set": 1.0},
        ),
    ],
    "powerlook": [
        ArchetypeConfig(
            slug="corporate",
            n_users=5,
            category_weights={"Shirt": 7.0, "Bottom": 3.0},
        ),
        ArchetypeConfig(
            slug="smartcasual",
            n_users=5,
            category_weights={"T-Shirt": 4.0, "Shirt": 6.0},
        ),
        ArchetypeConfig(
            slug="casual",
            n_users=5,
            category_weights={"T-Shirt": 7.0, "Vest": 3.0},
        ),
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_catalog(brand: str) -> pd.DataFrame:
    """Load a brand's catalog CSV and return it."""
    path = DATA_DIR / brand / "catalog.csv"
    df = pd.read_csv(path)
    return df


def build_category_pool(catalog: pd.DataFrame, weights: dict[str, float]) -> list[str]:
    """Return a weighted list of product_ids matching the archetype's categories.

    Categories with higher weights are repeated proportionally so that
    np.random.choice (without explicit p=) respects the desired mix.
    """
    frames: list[pd.DataFrame] = []
    for cat, w in weights.items():
        subset = catalog[catalog["category"] == cat][["product_id"]].copy()
        if subset.empty:
            continue
        # Repeat rows proportional to weight (rounded to nearest int, min 1).
        repeats = max(1, round(w))
        frames.append(pd.concat([subset] * repeats, ignore_index=True))
    if not frames:
        raise ValueError(f"No catalog items found for weights: {weights}")
    return pd.concat(frames, ignore_index=True)["product_id"].tolist()


def random_timestamps(n: int, rng: np.random.Generator) -> list[pd.Timestamp]:
    """Return n ascending timestamps uniformly drawn from the date range."""
    span_seconds = int((END_DATE - START_DATE).total_seconds())
    offsets = sorted(rng.integers(0, span_seconds, size=n).tolist())
    return [START_DATE + pd.Timedelta(seconds=int(s)) for s in offsets]


def generate_interactions(
    brand: str,
    archetypes: list[ArchetypeConfig],
    catalog: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate all synthetic interactions for a single brand."""
    rows: list[dict] = []

    for arch in archetypes:
        pool = build_category_pool(catalog, arch.category_weights)

        for i in range(1, arch.n_users + 1):
            user_id = f"synthetic_{brand}_{arch.slug}_{i:03d}"
            n_events = int(rng.integers(arch.interactions_range[0], arch.interactions_range[1] + 1))

            # Sample product_ids: without replacement if pool is large enough.
            pool_arr = np.array(pool)
            if len(pool_arr) >= n_events:
                chosen_indices = rng.choice(len(pool_arr), size=n_events, replace=False)
            else:
                chosen_indices = rng.choice(len(pool_arr), size=n_events, replace=True)
            product_ids = pool_arr[chosen_indices].tolist()

            timestamps = random_timestamps(n_events, rng)

            for pid, ts in zip(product_ids, timestamps):
                rows.append(
                    {
                        "user_id": user_id,
                        "product_id": pid,
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": EVENT_TYPE,
                    }
                )

    return pd.DataFrame(rows, columns=["user_id", "product_id", "timestamp", "event_type"])


def validate_product_ids(interactions: pd.DataFrame, catalog: pd.DataFrame, brand: str) -> None:
    """Assert every product_id in interactions exists in the catalog."""
    catalog_ids = set(catalog["product_id"].astype(str))
    interaction_ids = set(interactions["product_id"].astype(str))
    missing = interaction_ids - catalog_ids
    if missing:
        raise ValueError(
            f"[{brand}] {len(missing)} product_id(s) not found in catalog: "
            f"{sorted(missing)[:5]} ..."
        )


def print_summary(brand: str, interactions: pd.DataFrame) -> None:
    """Print per-brand summary stats."""
    n_users = interactions["user_id"].nunique()
    n_events = len(interactions)
    n_unique_products = interactions["product_id"].nunique()
    print(f"  users            : {n_users}")
    print(f"  total events     : {n_events}")
    print(f"  unique product_ids: {n_unique_products}")
    # Per-archetype breakdown
    interactions["archetype"] = interactions["user_id"].str.split("_").str[2]
    print("  events by archetype:")
    for arch, grp in interactions.groupby("archetype"):
        print(f"    {arch:15s}  users={grp['user_id'].nunique()}  events={len(grp)}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: generate and write synthetic interaction CSVs."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    for brand, archetypes in BRAND_CONFIG.items():
        print(f"\n=== {brand.upper()} ===")
        catalog = load_catalog(brand)
        interactions = generate_interactions(brand, archetypes, catalog, rng)
        validate_product_ids(interactions, catalog, brand)
        print_summary(brand, interactions)

        out_path = DATA_DIR / brand / "synthetic_users.csv"
        # Drop the temporary 'archetype' helper column before writing
        interactions.drop(columns=["archetype"], errors="ignore").to_csv(
            out_path, index=False
        )
        print(f"  written to       : {out_path}")

    print("\nAll brands processed successfully.")


if __name__ == "__main__":
    main()
