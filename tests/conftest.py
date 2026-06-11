from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rerank import RerankConfig


@pytest.fixture(scope="module")
def mock_art_map() -> dict[int, dict]:
    return {
        111: {
            "prod_name": "Blue Jeans",
            "colour_group_name": "Blue",
            "product_type_name": "Trousers",
        },
        222: {
            "prod_name": "Red Shirt",
            "colour_group_name": "Red",
            "product_type_name": "T-shirt",
        },
        333: {
            "prod_name": "Black Dress",
            "colour_group_name": "Black",
            "product_type_name": "Dress",
        },
    }


@pytest.fixture(scope="module")
def mock_brand_state(mock_art_map: dict) -> MagicMock:
    state = MagicMock()
    state.config.brand = "test_brand"
    state.config.display_name = "Test Brand"
    state.config.llm.enabled = False
    state.config.llm.provider = "template"
    state.api_key = "test-api-key-123"
    state.retriever.index.ntotal = 3
    state.retriever.search.return_value = [(111, 0.92), (222, 0.85), (333, 0.76)]
    state.retriever.index.reconstruct.side_effect = (
        lambda row: np.zeros(256, dtype=np.float32)
    )
    state.config.rerank = RerankConfig()
    state.art_map = mock_art_map
    state.faiss_aid_to_row = {111: 0, 222: 1, 333: 2}
    state.user_history = None
    state.device = torch.device("cpu")
    return state


@pytest.fixture(scope="module")
def mock_registry(mock_brand_state: MagicMock) -> MagicMock:
    registry = MagicMock()
    registry.get.side_effect = (
        lambda brand: mock_brand_state if brand == "test_brand" else None
    )
    registry.brand_names.return_value = ["test_brand"]
    return registry


@pytest.fixture(scope="module")
def api_client(mock_registry: MagicMock):
    from fastapi.testclient import TestClient

    with patch("app.api.main.load_registry", return_value=mock_registry):
        from app.api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
