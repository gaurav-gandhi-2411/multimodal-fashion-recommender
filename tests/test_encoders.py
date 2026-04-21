"""
Tests for ImageEncoder and TextEncoder.
Uses a real H&M image (article 698328004) to ensure the pipeline is exercised end-to-end,
not just on synthetic tensors.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

REAL_IMAGE_PATH = Path(
    "data/h-and-m-personalized-fashion-recommendations/images/069/0698328004.jpg"
)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)


@pytest.fixture(scope="module")
def img_encoder():
    from src.encoders.image_encoder import ImageEncoder
    return ImageEncoder(CFG)


@pytest.fixture(scope="module")
def txt_encoder():
    from src.encoders.text_encoder import TextEncoder
    return TextEncoder(CFG)


# --- Image encoder tests ---

def test_image_encoder_shape(img_encoder):
    emb = img_encoder.encode_batch([REAL_IMAGE_PATH])
    assert emb.shape == (1, 512), f"Expected (1, 512), got {emb.shape}"


def test_image_encoder_norm(img_encoder):
    emb = img_encoder.encode_batch([REAL_IMAGE_PATH])
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-5, f"Expected norm≈1.0, got {norm}"


def test_image_encoder_deterministic(img_encoder):
    emb1 = img_encoder.encode_batch([REAL_IMAGE_PATH])
    emb2 = img_encoder.encode_batch([REAL_IMAGE_PATH])
    assert np.allclose(emb1, emb2, atol=1e-6), "Image encoder is not deterministic"


def test_image_encoder_missing_file_returns_zero(img_encoder):
    missing = Path("data/nonexistent_image_99999.jpg")
    emb = img_encoder.encode_batch([missing])
    assert emb.shape == (1, 512)
    assert np.allclose(emb[0], 0.0), "Missing image should produce a zero vector"


# --- Text encoder tests ---

SAMPLE_TEXT = "Black slim-fit jeans. Trousers. Black. Five-pocket jeans in washed denim."


def test_text_encoder_shape(txt_encoder):
    emb = txt_encoder.encode_batch([SAMPLE_TEXT])
    assert emb.shape == (1, 384), f"Expected (1, 384), got {emb.shape}"


def test_text_encoder_norm(txt_encoder):
    emb = txt_encoder.encode_batch([SAMPLE_TEXT])
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-5, f"Expected norm≈1.0, got {norm}"


def test_text_encoder_deterministic(txt_encoder):
    emb1 = txt_encoder.encode_batch([SAMPLE_TEXT])
    emb2 = txt_encoder.encode_batch([SAMPLE_TEXT])
    assert np.allclose(emb1, emb2, atol=1e-6), "Text encoder is not deterministic"
