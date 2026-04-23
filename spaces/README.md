---
title: Multimodal Fashion Recommender
emoji: 👗
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
license: mit
---

# Multimodal Fashion Recommender

[![GitHub](https://img.shields.io/badge/GitHub-repo-black)](https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender/blob/main/LICENSE)

A two-tower retrieval system for personalised fashion recommendations, combining CLIP image embeddings, SBERT text embeddings, and a Transformer-based user sequence model. Per-recommendation modality breakdowns and one-sentence LLM explanations via Groq.

---

## How it works

**Item Tower**: frozen CLIP ViT-B/32 (512-dim) + SBERT all-MiniLM-L6 (384-dim) → concat → MLP → 256-dim unit-hypersphere. All item embeddings are pre-computed offline.

**User Tower**: Transformer (2 layers, 4 heads) over a user's last 20 item embeddings → masked mean-pool → MLP → 256-dim. Runs at request time (~5 ms on CPU).

**Retrieval**: FAISS `IndexFlatIP` over 1,500 top-frequency items. Dot product on unit vectors = cosine similarity.

**Explanations**: Groq API (LLaMA 3.1) generates a one-sentence rationale grounded in the user's visible style history.

---

## Demo users

30 pre-selected users across three buyer archetypes:

- **Specialist** — 80%+ of history in one product type (e.g. all Dresses)
- **Outfit Builder** — mixed product types with shared visual aesthetic
- **Colour / Aesthetic Buyer** — consistent colour palette across diverse product types

---

## Scores explained

Each recommendation card shows:

- **Similarity** — cosine distance in the 256-dim embedding space (green ≥ 0.52 / gray ≥ 0.45 / orange < 0.45)
- **Catalogue rank** — item's position when all 10,556 active items are ranked by similarity to this user
- **Img / Text** — modality-specific cosine scores (image-only vs text-only forward pass through ItemTower)
- **Confidence gap** — difference between rank-1 and rank-5 similarity; larger gap = clearer style signal

---

## Source

Full training code, pipeline, and results: [github.com/gaurav-gandhi-2411/multimodal-fashion-recommender](https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender)
