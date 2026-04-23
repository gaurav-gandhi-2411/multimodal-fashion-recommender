"""
Phase 4 index builder.
Loads best checkpoint, encodes all catalogue items through ItemTower -> 256-dim,
builds two FAISS IndexFlatIP indices:
  - full: all 20k catalogue items
  - active: items that appear in at least one transaction (train+val+test)

Saves:
  data/processed/faiss_index/         (full, 20k items)
  data/processed/faiss_index_active/  (active items only)
  data/processed/index_article_ids.npy         (article ids for full index)
  data/processed/index_article_ids_active.npy  (article ids for active index)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissRetriever
from src.training.train import encode_all_items


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "checkpoints/best.pt not found -- run scripts/02_train_model.py first."
        )

    # Load model from best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TwoTowerModel(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint['metrics']}")

    # Load raw precomputed embeddings
    processed = Path(config["data"]["processed_path"])
    img_emb  = np.load(processed / "item_image_embeddings.npy")
    txt_emb  = np.load(processed / "item_text_embeddings.npy")
    item_ids = np.load(processed / "item_ids_image.npy", allow_pickle=True)
    item_ids_int = [int(aid) for aid in item_ids]

    print(f"\nEncoding {len(item_ids):,} items through ItemTower (-> 256-dim)...")
    item_embs_256 = encode_all_items(model, img_emb, txt_emb, device, batch_size=512)
    print(f"  item_embs_256 shape: {item_embs_256.shape}")

    norms = np.linalg.norm(item_embs_256, axis=1)
    print(f"  L2 norm: mean={norms.mean():.6f}, std={norms.std():.6f}  (expect ~1.0, ~0.0)")

    # --- Full index (all catalogue items) ---
    article_ids_str = [str(aid) for aid in item_ids_int]
    retriever_full = FaissRetriever(item_embs_256, article_ids_str, metric="cosine")
    index_path = processed / "faiss_index"
    retriever_full.save(str(index_path))
    np.save(processed / "index_article_ids.npy", np.array(item_ids_int))
    print(f"\nFull index: {len(item_ids_int):,} items -> {index_path}")

    # --- Active items filter ---
    # Items that appear in at least one transaction across train+val+test
    print("\nLoading transaction splits to find active items...")
    train_df = pd.read_parquet(processed / "train.parquet")
    val_df   = pd.read_parquet(processed / "val.parquet")
    test_df  = pd.read_parquet(processed / "test.parquet")

    all_txn = pd.concat([train_df, val_df, test_df], ignore_index=True)
    active_article_ids = set(all_txn["article_id"].unique())
    print(f"  Total unique articles in transactions: {len(active_article_ids):,}")

    # Filter to items that are also in our embedding index
    catalogue_set = set(item_ids_int)
    active_in_catalogue = sorted(active_article_ids & catalogue_set)
    print(f"  Active articles in catalogue: {len(active_in_catalogue):,} "
          f"(of {len(item_ids_int):,} total)")

    # Build active index using same encoded embeddings (already computed)
    id_to_row = {int(aid): i for i, aid in enumerate(item_ids_int)}
    active_rows = np.array([id_to_row[aid] for aid in active_in_catalogue])
    active_embs = item_embs_256[active_rows]
    active_ids_str = [str(aid) for aid in active_in_catalogue]

    retriever_active = FaissRetriever(active_embs, active_ids_str, metric="cosine")
    index_path_active = processed / "faiss_index_active"
    retriever_active.save(str(index_path_active))
    np.save(processed / "index_article_ids_active.npy", np.array(active_in_catalogue))
    print(f"  Active index saved -> {index_path_active}")

    # Sanity check
    print("\nSanity check -- querying item[0] against full index:")
    results = retriever_full.search(item_embs_256[0], k=3)
    print(f"  Top-3: {results[:3]}")
    if results[0][0] == article_ids_str[0]:
        print(f"  PASSED: {article_ids_str[0]} is its own top-1 match.")
    else:
        print(f"  WARNING: top-1 is {results[0][0]}, expected {article_ids_str[0]}")

    print("\nBoth FAISS indices built and saved successfully.")
    print(f"  Full:   {len(item_ids_int):,} items")
    print(f"  Active: {len(active_in_catalogue):,} items")


if __name__ == "__main__":
    main()
