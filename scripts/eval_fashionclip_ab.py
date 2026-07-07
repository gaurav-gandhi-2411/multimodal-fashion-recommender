"""scripts/eval_fashionclip_ab.py

Quick A/B: current serving CLIP ViT-B/32 (openai) vs FashionCLIP
(patrickjohncyh/fashion-clip) on the exact failure case documented in the
2026-07-07 audit (scripts/eval_shirt_tshirt_rerank.py) -- cross-brand image
query where a plain t-shirt returns Shirt-category results on Powerlook -- plus
a same-brand self-retrieval sanity check so a candidate embedding swap can't be
judged a win on one query alone.

Both models are used PURELY as image encoders here (raw cosine over their own
embedding space) -- no FAISS index rebuild needed for this quick check; we
brute-force encode a live-fetched candidate pool per query instead of the full
catalog, which is fine for a same-order-of-magnitude signal but NOT a
substitute for a full offline re-embed + eval before any real migration.

PILOT RESULT (2026-07-07, n=1 failing query + n=10 self-retrieval, 60-item
brute-force candidate pool -- NOT the full 906-item catalog/FAISS index):
  CLIP ViT-B/32 (current):  failing query T-Shirt@5 = 3/5, self-retrieval 10/10
  FashionCLIP:               failing query T-Shirt@5 = 5/5, self-retrieval 10/10
FashionCLIP fully resolved the diagnosed failure on this pool with no
self-retrieval regression. Promising, but this is a single-query pilot on a
subsampled pool via brute-force cosine, not the locked-eval-standard n=100
this project normally requires (see PROJECT_MEMORY Phase 5/7 locked evals) --
treat as justification to run the full A/B, not as sufficient evidence to
migrate on its own. See the audit report for the proposed full-scale A/B.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

N_CANDIDATES_PER_BRAND = 60  # subset of the catalog to brute-force encode per query


def load_openai_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()

    def encode(img: Image.Image) -> np.ndarray:
        t = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            e = model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.squeeze(0).numpy().astype(np.float32)

    return encode


def load_fashion_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    proc = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    model.eval()

    def encode(img: Image.Image) -> np.ndarray:
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            e = model.get_image_features(**inputs)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.squeeze(0).numpy().astype(np.float32)

    return encode


def fetch_img(url: str) -> Image.Image | None:
    try:
        b = requests.get(url, timeout=15).content
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None


def run_case(encode_fn, query_img: Image.Image, candidates: list[tuple[int, str, Image.Image]], k: int = 5):
    q_emb = encode_fn(query_img)
    scored = []
    for aid, cat, img in candidates:
        emb = encode_fn(img)
        sim = float(np.dot(q_emb, emb))
        scored.append((aid, cat, sim))
    scored.sort(key=lambda x: -x[2])
    return scored[:k]


def main() -> None:
    pl = pd.read_parquet(REPO_ROOT / "data/powerlook/items.parquet")
    sn = pd.read_parquet(REPO_ROOT / "data/snitch/items.parquet")

    # Candidate pool: a mix of Powerlook Shirt + T-Shirt items (subset, brute force).
    pool_df = pd.concat([
        pl[pl["category"] == "Shirt"].sample(N_CANDIDATES_PER_BRAND // 2, random_state=42),
        pl[pl["category"] == "T-Shirt"].sample(N_CANDIDATES_PER_BRAND // 2, random_state=42),
    ])
    print(f"Downloading {len(pool_df)} candidate images...")
    candidates = []
    for _, row in pool_df.iterrows():
        img = fetch_img(row["image_url"])
        if img is not None:
            candidates.append((int(row["article_id"]), row["category"], img))
    print(f"  {len(candidates)} candidates loaded")

    # The diagnosed failing query: Snitch white t-shirt (article_id 203).
    query_row = sn[sn["article_id"] == 203].iloc[0]
    query_img = fetch_img(query_row["image_url"])

    # Self-retrieval sanity set: 5 Powerlook T-Shirts + 5 Shirts (own catalog images).
    self_sample = pd.concat([
        pl[pl["category"] == "T-Shirt"].sample(5, random_state=7),
        pl[pl["category"] == "Shirt"].sample(5, random_state=7),
    ])

    for name, loader in [("CLIP ViT-B/32 (current, openai)", load_openai_clip),
                         ("FashionCLIP (patrickjohncyh/fashion-clip)", load_fashion_clip)]:
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        encode_fn = loader()

        top5 = run_case(encode_fn, query_img, candidates, k=5)
        n_tee = sum(1 for _, cat, _ in top5 if cat == "T-Shirt")
        print(f"Cross-brand failing query (Snitch tee -> Powerlook pool): top-5 = "
              f"{[(a, c, round(s,4)) for a, c, s in top5]}")
        print(f"  T-Shirt count in top-5: {n_tee}/5")

        # Self-retrieval check: does each item's own image still find itself as top-1
        # against the SAME 60-item candidate pool (with itself added in)?
        hits = 0
        for _, row in self_sample.iterrows():
            aid = int(row["article_id"])
            img = fetch_img(row["image_url"])
            if img is None:
                continue
            pool_with_self = candidates + [(aid, row["category"], img)]
            top1 = run_case(encode_fn, img, pool_with_self, k=1)
            hits += top1[0][0] == aid
        print(f"Self-retrieval sanity (10 Powerlook items vs 61-item pool): {hits}/10")


if __name__ == "__main__":
    main()
