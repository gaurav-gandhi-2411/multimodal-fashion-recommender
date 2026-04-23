"""
Phase 4.1 baselines.

BASELINE A -- Item popularity
  Count article occurrences in train_df, rank descending.
  Evaluate Recall@10 and NDCG@10 on val and test by recommending the
  same top-10 to every user.

BASELINE B -- Text-only two-tower
  Same model architecture but image embeddings are zeroed out.
  Train for 5 epochs; save checkpoint to checkpoints/text_only.pt.
  Evaluate on val and test (full catalogue).

Outputs: prints per-split metrics for both baselines.
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
from src.training.evaluate import (
    ndcg_at_k,
    popularity_ndcg_at_k,
    popularity_recall_at_k,
    recall_at_k,
)
from src.training.train import encode_all_items, run_sanity_check, train, _collect_user_embs
from torch.utils.data import DataLoader

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

    img_emb  = np.load(processed / "item_image_embeddings.npy")
    txt_emb  = np.load(processed / "item_text_embeddings.npy")
    item_ids = np.load(processed / "item_ids_image.npy", allow_pickle=True)
    article_id_to_idx = {int(aid): i for i, aid in enumerate(item_ids)}

    train_df = pd.read_parquet(processed / "train.parquet")
    val_df   = pd.read_parquet(processed / "val.parquet")
    test_df  = pd.read_parquet(processed / "test.parquet")

    seq_len = config["model"]["user_seq_len"]
    bs      = config["training"]["batch_size"]

    # ------------------------------------------------------------------ #
    # BASELINE A: Item popularity                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BASELINE A: Item popularity")
    print("=" * 60)

    # Rank articles by frequency in train_df
    counts = train_df["article_id"].value_counts()
    popular_article_ids = counts.index.tolist()   # sorted descending by count
    popular_item_indices = [
        article_id_to_idx[aid]
        for aid in popular_article_ids
        if aid in article_id_to_idx
    ]
    print(f"  Unique articles in train: {len(popular_item_indices):,}")
    print(f"  Top-3 popular item indices: {popular_item_indices[:3]}")

    # Collect true item indices for val and test
    def get_true_indices(targets_df: pd.DataFrame) -> np.ndarray:
        mask = targets_df["article_id"].isin(article_id_to_idx)
        return np.array([
            article_id_to_idx[aid]
            for aid in targets_df.loc[mask, "article_id"]
        ])

    val_true_idx  = get_true_indices(val_df)
    test_true_idx = get_true_indices(test_df)

    pop_val_recall  = popularity_recall_at_k(val_true_idx,  popular_item_indices, k=10)
    pop_val_ndcg    = popularity_ndcg_at_k(val_true_idx,    popular_item_indices, k=10)
    pop_test_recall = popularity_recall_at_k(test_true_idx, popular_item_indices, k=10)
    pop_test_ndcg   = popularity_ndcg_at_k(test_true_idx,   popular_item_indices, k=10)

    print(f"  Val  | Recall@10={pop_val_recall:.4f} | NDCG@10={pop_val_ndcg:.4f}")
    print(f"  Test | Recall@10={pop_test_recall:.4f} | NDCG@10={pop_test_ndcg:.4f}")

    # ------------------------------------------------------------------ #
    # BASELINE B: Text-only two-tower                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BASELINE B: Text-only two-tower (zeroed image embeddings)")
    print("=" * 60)

    # Zero out image embeddings -- same array shape, all zeros
    zero_img_emb = np.zeros_like(img_emb)

    # Datasets
    full_hist_val  = pd.concat([train_df, val_df],           ignore_index=True)
    full_hist_test = pd.concat([train_df, val_df, test_df],  ignore_index=True)

    print("\nBuilding train dataset (text-only)...")
    train_dataset = FashionInteractionDataset(
        interactions_df=train_df,
        item_img_emb=zero_img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
    )
    print("\nBuilding val dataset (text-only)...")
    val_dataset = FashionInteractionDataset(
        interactions_df=full_hist_val,
        item_img_emb=zero_img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
        targets_df=val_df,
    )
    print("\nBuilding test dataset (text-only)...")
    test_dataset = FashionInteractionDataset(
        interactions_df=full_hist_test,
        item_img_emb=zero_img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
        targets_df=test_df,
    )

    # Train text-only for 5 epochs
    text_config = {k: v for k, v in config.items()}
    text_config["training"] = dict(config["training"])
    text_config["training"]["num_epochs"] = 5

    model = TwoTowerModel(text_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    run_sanity_check(model, train_dataset, device)

    Path("checkpoints").mkdir(exist_ok=True)
    best_metrics = train(
        config=text_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        all_img_emb=zero_img_emb,
        all_txt_emb=txt_emb,
        device=device,
        test_dataset=test_dataset,
        checkpoint_path="checkpoints/text_only.pt",
    )

    print("\n-- Text-only final metrics --")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY -- full catalogue (20k items)")
    print("=" * 60)
    print(f"  Popularity  | val Recall@10={pop_val_recall:.4f} NDCG@10={pop_val_ndcg:.4f}"
          f"  | test Recall@10={pop_test_recall:.4f} NDCG@10={pop_test_ndcg:.4f}")
    txt_vr = best_metrics.get("val_recall_at_10",  "N/A")
    txt_vn = best_metrics.get("val_ndcg_at_10",    "N/A")
    txt_tr = best_metrics.get("test_recall_at_10", "N/A")
    txt_tn = best_metrics.get("test_ndcg_at_10",   "N/A")
    print(f"  Text-only   | val Recall@10={txt_vr:.4f} NDCG@10={txt_vn:.4f}"
          f"  | test Recall@10={txt_tr:.4f} NDCG@10={txt_tn:.4f}")

    # Save popularity metrics for comparison script to load
    np.save(
        processed / "popularity_metrics.npy",
        np.array([pop_val_recall, pop_val_ndcg, pop_test_recall, pop_test_ndcg]),
    )
    print("\nPopularity metrics saved to data/processed/popularity_metrics.npy")


if __name__ == "__main__":
    main()
