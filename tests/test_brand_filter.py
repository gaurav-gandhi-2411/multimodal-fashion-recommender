from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.brands.registry import _enabled_brands, load_registry


# ---------------------------------------------------------------------------
# Unit tests for _enabled_brands()
# ---------------------------------------------------------------------------


def test_enabled_brands_unset_returns_none() -> None:
    """_enabled_brands returns None when BRANDS_ENABLED is not in env (all brands)."""
    env_without = {k: v for k, v in os.environ.items() if k != "BRANDS_ENABLED"}
    with patch.dict(os.environ, env_without, clear=True):
        assert _enabled_brands() is None


def test_enabled_brands_empty_string_returns_none() -> None:
    """_enabled_brands returns None for empty string (all brands)."""
    with patch.dict(os.environ, {"BRANDS_ENABLED": ""}, clear=False):
        assert _enabled_brands() is None


def test_enabled_brands_whitespace_only_returns_none() -> None:
    """_enabled_brands returns None for whitespace-only value."""
    with patch.dict(os.environ, {"BRANDS_ENABLED": "  "}, clear=False):
        assert _enabled_brands() is None


def test_enabled_brands_simple_list() -> None:
    """_enabled_brands parses a comma-separated list into a set of lowercased slugs."""
    with patch.dict(os.environ, {"BRANDS_ENABLED": "a,b"}, clear=False):
        assert _enabled_brands() == {"a", "b"}


def test_enabled_brands_strips_whitespace_and_lowercases() -> None:
    """_enabled_brands strips whitespace and normalises to lowercase."""
    with patch.dict(os.environ, {"BRANDS_ENABLED": " A , b "}, clear=False):
        assert _enabled_brands() == {"a", "b"}


def test_enabled_brands_single_slug() -> None:
    """_enabled_brands works for a single slug with no commas."""
    with patch.dict(os.environ, {"BRANDS_ENABLED": "snitch"}, clear=False):
        assert _enabled_brands() == {"snitch"}


# ---------------------------------------------------------------------------
# Registry filter tests via load_registry()
# The key invariant: a brand NOT in BRANDS_ENABLED must never trigger _load_brand,
# so its data files are never touched.
# ---------------------------------------------------------------------------


def _minimal_yaml(slug: str) -> str:
    return f"""
brand: {slug}
display_name: {slug.title()}
catalog_path: data/{slug}/catalog.parquet
index_path: indices/{slug}/active.faiss
checkpoint_path: checkpoints/{slug}/best.pt
api_key_env: {slug.upper()}_API_KEY
"""


def test_load_registry_filter_skips_disabled_brand(tmp_path: Path) -> None:
    """load_registry with BRANDS_ENABLED only calls _load_brand for the named brand.

    The disabled brand's YAML is present on disk but _load_brand must never be
    invoked for it — proving the filter fires before any file I/O.
    """
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    (brands_dir / "alpha.yaml").write_text(_minimal_yaml("alpha"))
    (brands_dir / "beta.yaml").write_text(_minimal_yaml("beta"))

    mock_alpha_state = MagicMock()
    mock_alpha_state.config.brand = "alpha"

    # Patch _load_brand so we can track which slugs it was called with,
    # and avoid needing real parquet/faiss/checkpoint files on disk.
    with (
        patch.dict(os.environ, {"BRANDS_ENABLED": "alpha"}, clear=False),
        patch("app.brands.registry._load_brand", return_value=mock_alpha_state) as mock_load,
    ):
        registry = load_registry(str(brands_dir))

    # _load_brand must only be called once — for alpha, not for beta
    assert mock_load.call_count == 1
    called_path: Path = mock_load.call_args[0][0]
    assert called_path.stem == "alpha", f"Expected 'alpha.yaml', got {called_path.name!r}"

    # Registry contains only alpha
    assert registry.brand_names() == ["alpha"]


def test_load_registry_filter_loads_all_when_unset(tmp_path: Path) -> None:
    """load_registry loads all brands when BRANDS_ENABLED is not set."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    (brands_dir / "alpha.yaml").write_text(_minimal_yaml("alpha"))
    (brands_dir / "beta.yaml").write_text(_minimal_yaml("beta"))

    def fake_load(path: Path) -> MagicMock:
        state = MagicMock()
        state.config.brand = path.stem
        return state

    env_without = {k: v for k, v in os.environ.items() if k != "BRANDS_ENABLED"}
    with (
        patch.dict(os.environ, env_without, clear=True),
        patch("app.brands.registry._load_brand", side_effect=fake_load) as mock_load,
    ):
        registry = load_registry(str(brands_dir))

    assert mock_load.call_count == 2
    assert set(registry.brand_names()) == {"alpha", "beta"}


def test_load_registry_raises_when_filter_matches_nothing(tmp_path: Path) -> None:
    """load_registry raises RuntimeError when BRANDS_ENABLED matches no YAML brand."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    (brands_dir / "alpha.yaml").write_text(_minimal_yaml("alpha"))

    with (
        patch.dict(os.environ, {"BRANDS_ENABLED": "nonexistent"}, clear=False),
        pytest.raises(RuntimeError, match="BRANDS_ENABLED"),
    ):
        load_registry(str(brands_dir))
