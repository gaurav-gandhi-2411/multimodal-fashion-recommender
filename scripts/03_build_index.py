"""
Phase 4 index builder.
Loads best checkpoint, encodes all catalogue items through ItemTower -> 256-dim,
builds a FAISS IndexFlatIP, and saves to data/processed/faiss_index/.
"""
import sys
from pathlib import Path

import numpy as np
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
    article_ids = [str(aid) for aid in item_ids]

    print(f"Encoding {len(item_ids):,} items through ItemTower (-> 256-dim)...")
    item_embs_256 = encode_all_items(model, img_emb, txt_emb, device, batch_size=512)
    print(f"  item_embs_256 shape: {item_embs_256.shape}")

    # Verify L2 normalisation (ItemTower applies F.normalize in forward)
    norms = np.linalg.norm(item_embs_256, axis=1)
    print(f"  L2 norm: mean={norms.mean():.6f}, std={norms.std():.6f}  (expect ~1.0, ~0.0)")

    # Build and save FAISS index
    retriever = FaissRetriever(item_embs_256, article_ids, metric="cosine")
    index_path = processed / "faiss_index"
    retriever.save(str(index_path))

    # Sanity check: an item should be its own top-1
    print("\nSanity check -- querying item[0] with its own embedding:")
    results = retriever.search(item_embs_256[0], k=10)
    print(f"  Top-3: {results[:3]}")
    top1_id = results[0][0]
    if top1_id == article_ids[0]:
        print(f"  PASSED: {article_ids[0]} is its own top-1 match.")
    else:
        print(f"  WARNING: top-1 is {top1_id}, expected {article_ids[0]}")

    print("\nFAISS index built and saved successfully.")


if __name__ == "__main__":
    main()
