from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.rerank import RerankConfig
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissRetriever


def _enabled_brands() -> set[str] | None:
    """
    Return the set of lower-cased brand slugs that should be loaded, or None when
    BRANDS_ENABLED is unset/empty (meaning "all brands").

    Env var format: comma-separated slugs, e.g. ``BRANDS_ENABLED=snitch,fashor``.
    Whitespace around each slug is stripped; empty tokens are dropped.
    """
    raw = os.environ.get("BRANDS_ENABLED", "")
    if not raw.strip():
        return None
    brands = {slug.strip().lower() for slug in raw.split(",") if slug.strip()}
    return brands if brands else None


class LLMBrandConfig(BaseModel):
    provider: str = "template"
    model: str = "llama-3.1-8b-instant"
    enabled: bool = True


class BrandConfig(BaseModel):
    brand: str
    display_name: str
    catalog_path: str
    index_path: str
    transactions_dir: str | None = None
    checkpoint_path: str = "checkpoints/best.pt"
    api_key_env: str
    embeddings_path: str | None = None
    pdp_url_template: str | None = None
    llm: LLMBrandConfig = Field(default_factory=LLMBrandConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


@dataclass
class BrandState:
    config: BrandConfig
    catalog: pd.DataFrame
    art_map: dict[int, dict]
    retriever: FaissRetriever
    faiss_aid_to_row: dict[int, int]
    model: TwoTowerModel
    device: torch.device
    user_history: pd.DataFrame | None
    api_key: str


class BrandRegistry:
    def __init__(self) -> None:
        self._brands: dict[str, BrandState] = {}

    def register(self, state: BrandState) -> None:
        self._brands[state.config.brand] = state

    def get(self, brand: str) -> BrandState | None:
        return self._brands.get(brand)

    def brand_names(self) -> list[str]:
        return list(self._brands.keys())

    def __len__(self) -> int:
        return len(self._brands)


def _load_brand(yaml_path: Path) -> BrandState:
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    config = BrandConfig.model_validate(data)

    catalog = pd.read_parquet(config.catalog_path)
    catalog["article_id"] = catalog["article_id"].astype(int)
    art_map: dict[int, dict] = catalog.set_index("article_id").to_dict("index")

    retriever = FaissRetriever.load(config.index_path)
    faiss_aid_to_row: dict[int, int] = {
        int(aid): i for i, aid in enumerate(retriever.article_ids)
    }

    if config.transactions_dir:
        td = Path(config.transactions_dir)
        splits = [
            pd.read_parquet(td / f"{split}.parquet")
            for split in ("train", "val", "test")
        ]
        history = pd.concat(splits, ignore_index=True)
        history["article_id"] = history["article_id"].astype(int)
        history = history.sort_values("t_dat")
    else:
        history = None

    device = torch.device("cpu")
    ckpt = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Brand {config.brand!r}: env var {config.api_key_env!r} is not set. "
            "Set it before starting the server."
        )

    return BrandState(
        config=config,
        catalog=catalog,
        art_map=art_map,
        retriever=retriever,
        faiss_aid_to_row=faiss_aid_to_row,
        model=model,
        device=device,
        user_history=history,
        api_key=api_key,
    )


def load_registry(brands_dir: str | Path = "brands") -> BrandRegistry:
    """
    Load and return a BrandRegistry from all YAML files in *brands_dir*.

    When the ``BRANDS_ENABLED`` env var is set (comma-separated brand slugs), only
    brands whose ``brand:`` field matches an enabled slug are loaded; others are skipped
    without reading their data files.  When the env var is unset or empty, all brands
    are loaded (preserving existing behaviour).

    Raises RuntimeError if no YAML files are found, or if BRANDS_ENABLED is set but
    filters out every brand (likely misconfiguration).
    """
    enabled = _enabled_brands()
    registry = BrandRegistry()
    brands_path = Path(brands_dir)
    yaml_files = sorted(brands_path.glob("*.yaml"))
    if not yaml_files:
        raise RuntimeError(f"No brand YAML files found in {brands_path}")
    for yaml_path in yaml_files:
        if enabled is not None:
            with yaml_path.open() as f:
                slug = yaml.safe_load(f).get("brand", "").lower()
            if slug not in enabled:
                continue
        state = _load_brand(yaml_path)
        registry.register(state)
    if len(registry) == 0:
        raise RuntimeError(
            f"BRANDS_ENABLED={os.environ.get('BRANDS_ENABLED')!r} filtered out every brand. "
            "Check the spelling of brand slugs against the 'brand:' field in each YAML."
        )
    return registry
