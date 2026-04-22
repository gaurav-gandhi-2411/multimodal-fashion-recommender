"""
Phase 4 training orchestrator.
Loads precomputed embeddings, builds datasets, runs sanity check, then full training.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.training.dataset import FashionInteractionDataset
from src.training.train import run_sanity_check, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processed = Path(config["data"]["processed_path"])

    # ── Load precomputed embeddings (canonical source: item_ids_image.npy) ──
    img_emb  = np.load(processed / "item_image_embeddings.npy")
    txt_emb  = np.load(processed / "item_text_embeddings.npy")
    item_ids = np.load(processed / "item_ids_image.npy", allow_pickle=True)

    assert img_emb.shape[0] == txt_emb.shape[0] == len(item_ids), (
        f"Embedding/id count mismatch: img={img_emb.shape[0]}, "
        f"txt={txt_emb.shape[0]}, ids={len(item_ids)}"
    )
    print(f"Embeddings: img={img_emb.shape}, txt={txt_emb.shape}, items={len(item_ids):,}")

    # Single source of truth: article_id → row index in embedding matrices.
    # Use int keys to match the int64 dtype in the parquet interaction files.
    article_id_to_idx = {int(aid): i for i, aid in enumerate(item_ids)}

    # ── Load interaction splits ───────────────────────────────────────────────
    train_df = pd.read_parquet(processed / "train.parquet")
    val_df   = pd.read_parquet(processed / "val.parquet")
    print(f"Interactions — train: {len(train_df):,}, val: {len(val_df):,}")

    seq_len = config["model"]["user_seq_len"]

    train_dataset = FashionInteractionDataset(
        interactions_df=train_df,
        item_img_emb=img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
    )
    val_dataset = FashionInteractionDataset(
        interactions_df=val_df,
        item_img_emb=img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
    )
    print(f"Samples — train: {len(train_dataset):,}, val: {len(val_dataset):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TwoTowerModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    run_sanity_check(model, train_dataset, device)

    # ── Full training ─────────────────────────────────────────────────────────
    best = train(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        all_img_emb=img_emb,
        all_txt_emb=txt_emb,
        device=device,
    )

    print("\n── Final best checkpoint metrics ──")
    for k, v in best.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
