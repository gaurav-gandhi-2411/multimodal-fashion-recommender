import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

processed_dir = Path(cfg["data"]["processed_path"])
images_dir = Path(cfg["data"]["images_dir"])

articles = pd.read_parquet(processed_dir / "articles.parquet")
print(f"Articles to encode: {len(articles):,}")

# --- Image embeddings ---
print("\n=== Image Encoding (CLIP ViT-B/32) ===")
img_encoder = ImageEncoder(cfg)
img_out = processed_dir / "item_image_embeddings.npy"

t0 = time.time()
img_embs, img_ids = img_encoder.encode_directory(
    articles,
    images_dir=images_dir,
    output_path=img_out,
    batch_size=cfg["encoders"]["batch_size"],
    num_workers=4,
)
img_time = time.time() - t0

print(f"Image encoding time: {img_time:.1f}s ({img_time/60:.1f} min)")
print(f"Saved: {img_out}  shape={img_embs.shape}")
norms = np.linalg.norm(img_embs, axis=1)
# Zero-vectors (missing images) have norm 0; exclude from norm check
nonzero = norms[norms > 0]
print(f"Norm check (non-zero rows): mean={nonzero.mean():.6f}  min={nonzero.min():.6f}  max={nonzero.max():.6f}")

# --- Text embeddings ---
print("\n=== Text Encoding (SBERT all-MiniLM-L6-v2) ===")
txt_encoder = TextEncoder(cfg)
txt_out = processed_dir / "item_text_embeddings.npy"

t0 = time.time()
txt_embs = txt_encoder.encode_dataframe(
    articles,
    text_col="full_text",
    output_path=txt_out,
    batch_size=128,
)
txt_time = time.time() - t0

print(f"Text encoding time: {txt_time:.1f}s ({txt_time/60:.1f} min)")
print(f"Saved: {txt_out}  shape={txt_embs.shape}")
norms = np.linalg.norm(txt_embs, axis=1)
print(f"Norm check: mean={norms.mean():.6f}  min={norms.min():.6f}  max={norms.max():.6f}")

print("\nDone.")
