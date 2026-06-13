from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.complete import CompleteConfig
from app.rerank import RerankConfig
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissRetriever


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
    complete: CompleteConfig = Field(default_factory=CompleteConfig)


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
    # Prebuilt embedding matrix for O(1) lookup by FAISS row index.
    # Shape: (n_items, emb_dim) float32. Populated in _load_brand via reconstruct_n.
    item_embeddings: np.ndarray | None = field(default=None)
    # Inverse of faiss_aid_to_row: maps row index -> article_id (int).
    faiss_row_to_aid: dict[int, int] | None = field(default=None)


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

    # Preload the entire embedding matrix for O(1) candidate scoring in complete_the_look.
    # IndexFlatIP stores the original L2-normalised vectors verbatim; reconstruct_n returns
    # them in FAISS row order (same ordering as faiss_aid_to_row values).
    n_total = retriever.index.ntotal
    item_embeddings: np.ndarray = retriever.index.reconstruct_n(0, n_total).astype(np.float32)
    faiss_row_to_aid: dict[int, int] = {row: int(aid) for aid, row in faiss_aid_to_row.items()}

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
        item_embeddings=item_embeddings,
        faiss_row_to_aid=faiss_row_to_aid,
    )


def load_registry(brands_dir: str | Path = "brands") -> BrandRegistry:
    registry = BrandRegistry()
    brands_path = Path(brands_dir)
    yaml_files = sorted(brands_path.glob("*.yaml"))
    if not yaml_files:
        raise RuntimeError(f"No brand YAML files found in {brands_path}")
    for yaml_path in yaml_files:
        state = _load_brand(yaml_path)
        registry.register(state)
    return registry
