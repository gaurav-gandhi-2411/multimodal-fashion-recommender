"""
Tests for ItemTower, UserTower, TwoTowerModel, and FashionInteractionDataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

# Minimal config for isolated model tests (avoids loading full dataset)
MINI_CFG = {
    "encoders": {"image_embed_dim": 512, "text_embed_dim": 384},
    "model": {"item_fusion_hidden": 512, "output_dim": 256, "user_seq_len": 20, "dropout": 0.0},
    "training": {"temperature": 0.07},
}

BATCH = 4
SEQ_LEN = 20
IMG_DIM = 512
TXT_DIM = 384
OUT_DIM = 256

# Real user with >=5 training interactions (confirmed in Phase 3 setup)
TEST_USER_ID = "00009d946eec3ea54add5ba56d5210ea898def4b46c68570cf0096d962cacc75"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def item_tower():
    from src.models.item_tower import ItemTower
    return ItemTower(
        image_dim=IMG_DIM, text_dim=TXT_DIM,
        hidden=512, output_dim=OUT_DIM, dropout=0.0
    ).eval()


@pytest.fixture(scope="module")
def user_tower():
    from src.models.user_tower import UserTower
    return UserTower(item_dim=OUT_DIM, max_seq=SEQ_LEN, dropout=0.0).eval()


@pytest.fixture(scope="module")
def two_tower():
    from src.models.two_tower import TwoTowerModel
    return TwoTowerModel(MINI_CFG).eval()


@pytest.fixture(scope="module")
def dataset():
    from src.training.dataset import FashionInteractionDataset
    img_emb = np.load("data/processed/item_image_embeddings.npy")
    txt_emb = np.load("data/processed/item_text_embeddings.npy")
    img_ids = np.load("data/processed/item_ids_image.npy")
    article_to_idx = {int(aid): i for i, aid in enumerate(img_ids)}
    train_df = pd.read_parquet("data/processed/train.parquet")
    return FashionInteractionDataset(
        interactions_df=train_df,
        item_img_emb=img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_to_idx,
        seq_len=CFG["model"]["user_seq_len"],
    )


# ---------------------------------------------------------------------------
# ItemTower tests
# ---------------------------------------------------------------------------

def test_item_tower_output_shape(item_tower):
    img = torch.randn(BATCH, IMG_DIM)
    txt = torch.randn(BATCH, TXT_DIM)
    out = item_tower(img, txt)
    assert out.shape == (BATCH, OUT_DIM), f"Expected ({BATCH}, {OUT_DIM}), got {out.shape}"


def test_item_tower_unit_norm(item_tower):
    img = torch.randn(BATCH, IMG_DIM)
    txt = torch.randn(BATCH, TXT_DIM)
    out = item_tower(img, txt)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), f"Norms not ≈1: {norms}"


# ---------------------------------------------------------------------------
# UserTower tests
# ---------------------------------------------------------------------------

def test_user_tower_output_shape_no_mask(user_tower):
    seq = torch.randn(BATCH, SEQ_LEN, OUT_DIM)
    out = user_tower(seq)
    assert out.shape == (BATCH, OUT_DIM)


def test_user_tower_unit_norm_no_mask(user_tower):
    seq = torch.randn(BATCH, SEQ_LEN, OUT_DIM)
    out = user_tower(seq)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


def test_user_tower_output_shape_with_mask(user_tower):
    seq = torch.randn(BATCH, SEQ_LEN, OUT_DIM)
    # Simulate left-padded sequences: first 5 slots are padding, rest are real
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, 5:] = True
    out = user_tower(seq, attention_mask=mask)
    assert out.shape == (BATCH, OUT_DIM)


def test_user_tower_unit_norm_with_mask(user_tower):
    seq = torch.randn(BATCH, SEQ_LEN, OUT_DIM)
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, 5:] = True
    out = user_tower(seq, attention_mask=mask)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# TwoTowerModel tests
# ---------------------------------------------------------------------------

def test_two_tower_logits_shape(two_tower):
    user_seq_img = torch.randn(BATCH, SEQ_LEN, IMG_DIM)
    user_seq_txt = torch.randn(BATCH, SEQ_LEN, TXT_DIM)
    user_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
    target_img = torch.randn(BATCH, IMG_DIM)
    target_txt = torch.randn(BATCH, TXT_DIM)

    logits = two_tower(user_seq_img, user_seq_txt, user_mask, target_img, target_txt)
    assert logits.shape == (BATCH, BATCH), f"Expected ({BATCH}, {BATCH}), got {logits.shape}"


# ---------------------------------------------------------------------------
# FashionInteractionDataset tests
# ---------------------------------------------------------------------------

def test_dataset_nonempty(dataset):
    assert len(dataset) > 0, "Dataset has no samples"


def test_dataset_getitem_keys_and_shapes(dataset):
    sample = dataset[0]
    assert set(sample.keys()) == {
        "user_seq_img", "user_seq_txt", "user_mask", "target_img", "target_txt"
    }, f"Unexpected keys: {set(sample.keys())}"
    assert sample["user_seq_img"].shape == (SEQ_LEN, IMG_DIM)
    assert sample["user_seq_txt"].shape == (SEQ_LEN, TXT_DIM)
    assert sample["user_mask"].shape == (SEQ_LEN,)
    assert sample["target_img"].shape == (IMG_DIM,)
    assert sample["target_txt"].shape == (TXT_DIM,)


def test_dataset_getitem_dtypes(dataset):
    sample = dataset[0]
    assert sample["user_seq_img"].dtype == torch.float32
    assert sample["user_seq_txt"].dtype == torch.float32
    assert sample["user_mask"].dtype == torch.bool
    assert sample["target_img"].dtype == torch.float32
    assert sample["target_txt"].dtype == torch.float32


def test_dataset_getitem_mask_has_real_items(dataset):
    # Find a sample for our known test user that has prior history
    train_df = pd.read_parquet("data/processed/train.parquet")
    img_ids = np.load("data/processed/item_ids_image.npy")
    article_to_idx = {int(aid): i for i, aid in enumerate(img_ids)}

    # Look for any sample index belonging to the test user
    found_idx = None
    for i, (uid, _, _) in enumerate(dataset._samples):
        if uid == TEST_USER_ID:
            found_idx = i
            break

    assert found_idx is not None, f"Test user {TEST_USER_ID} not found in dataset samples"
    sample = dataset[found_idx]
    assert sample["user_mask"].any(), "Expected at least one real item in history mask"
