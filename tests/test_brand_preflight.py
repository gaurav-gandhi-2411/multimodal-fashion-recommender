"""tests/test_brand_preflight.py -- unit tests for scripts/brand_preflight.py's main().

No model loading, no FAISS, no network I/O — only YAML validation + filesystem existence
checks, exercised against tmp_path fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from brand_preflight import main  # noqa: E402


def _write_brand_yaml(brands_dir: Path, slug: str) -> None:
    """Write a minimal, valid brand YAML with the given slug into brands_dir."""
    content = f"""
brand: {slug}
display_name: {slug.title()}
catalog_path: data/{slug}/items.parquet
index_path: indices/{slug}/active.faiss
checkpoint_path: checkpoints/{slug}/best.pt
api_key_env: {slug.upper()}_API_KEY
"""
    (brands_dir / f"{slug}.yaml").write_text(content)


def test_main_all_present_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every required asset exists locally, main() prints all_present=True and exits 0."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    _write_brand_yaml(brands_dir, "testbrand")

    output_base = tmp_path / "assets"
    for rel in (
        "data/testbrand/items.parquet",
        "indices/testbrand/active.faiss/faiss.index",
        "indices/testbrand/active.faiss/article_ids.pkl",
        "checkpoints/testbrand/best.pt",
    ):
        p = output_base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stub")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "brand_preflight.py",
            "--brand",
            "testbrand",
            "--brands-dir",
            str(brands_dir),
            "--output-base",
            str(output_base),
        ],
    )

    exit_code = main()
    assert exit_code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert result["brand"] == "testbrand"
    assert result["missing_local"] == []
    assert result["all_present"] is True


def test_main_missing_assets_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When required assets are absent locally, main() reports them and exits 1."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    _write_brand_yaml(brands_dir, "testbrand")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "brand_preflight.py",
            "--brand",
            "testbrand",
            "--brands-dir",
            str(brands_dir),
            "--output-base",
            str(tmp_path / "empty_output_base"),
        ],
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert result["brand"] == "testbrand"
    assert result["all_present"] is False
    assert len(result["missing_local"]) == len(result["required_paths"])


def test_main_missing_yaml_file_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() exits 1 (no JSON) when the brand YAML file itself does not exist."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "brand_preflight.py",
            "--brand",
            "doesnotexist",
            "--brands-dir",
            str(tmp_path),
        ],
    )
    assert main() == 1


def test_main_malformed_yaml_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() exits 1 when the brand YAML fails BrandConfig validation (missing required fields)."""
    brands_dir = tmp_path / "brands"
    brands_dir.mkdir()
    # Missing required fields (catalog_path, index_path, api_key_env) -> ValidationError.
    (brands_dir / "broken.yaml").write_text("brand: broken\ndisplay_name: Broken\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["brand_preflight.py", "--brand", "broken", "--brands-dir", str(brands_dir)],
    )
    assert main() == 1
