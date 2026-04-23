# Multimodal Fashion Recommender with LLM-Generated Explanations

> A two-tower recommender combining CLIP image embeddings, sentence-transformer text encoders, and session-aware sequence modelling — with an LLM reasoning layer that generates natural-language explanations for each recommendation.

**[Live Demo on HuggingFace Spaces](#)** *(coming soon)* · **[Demo screenshots](#demo)**

---

## Motivation

Modern recommender systems are accurate but opaque. Users and product teams both benefit from knowing *why* an item is recommended. This project explores a practical way to combine retrieval-quality two-tower models with LLM-based post-hoc explanations, specifically for fashion e-commerce — using consumer hardware (RTX 3070 laptop) and fully open-source tooling.

---

## Architecture

```
Catalogue ──────────────────────────────────────────────────────────────────────
  article image  ──→  CLIP ViT-B/32 (frozen)  ──→  512-dim ─┐
                                                               ├─→  ItemTower MLP  ──→  256-dim (L2-norm)
  article text   ──→  SBERT all-MiniLM-L6 (frozen) ──→ 384-dim ─┘

User ────────────────────────────────────────────────────────────────────────────
  last 20 items  ──→  ItemTower (shared weights)  ──→  (20 × 256)
                 ──→  Transformer encoder (2L, 4H, mean-pool)
                 ──→  UserTower  ──→  256-dim (L2-norm)

Retrieval ───────────────────────────────────────────────────────────────────────
  user @ item matrix  ──→  FAISS IndexFlatIP (cosine on unit vecs)  ──→  top-K

Explanation ─────────────────────────────────────────────────────────────────────
  (user history, recommended item)  ──→  LLaMA 3.1 8B  ──→  1-sentence rationale
```

### Key design decisions

- **Two-tower architecture** — user and item towers are encoded independently, enabling pre-computation of the full item index. Only user encoding happens at request time.
- **Frozen CLIP + SBERT** — keeps training cheap, avoids catastrophic forgetting of visual/textual knowledge from billion-parameter pre-training.
- **Late fusion** — concatenate 512-dim image + 384-dim text → project to 256-dim. Lets the model learn which modality to weight per item type.
- **In-batch InfoNCE loss (τ=0.1)** — no explicit negative mining needed; every other item in the batch is a negative. Diagonal of the logit matrix = positive pairs.
- **Temporal train/val/test split** — chronological split prevents the leakage that inflates metrics in random-split recommender benchmarks.
- **Shallow Transformer user tower** — 2-layer transformer encoder with positional embeddings and masked mean-pooling captures sequence order without overfitting on the limited per-user history.
- **Post-retrieval LLM explanation** — LLM is called K times on already-retrieved items, not during ranking. Keeps latency proportional to K, not catalogue size.

---

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 105k articles, 31M transactions, 1.4M customers across 2 years of purchase history.

This project uses a **20k-article, 2M-transaction sample** for training tractability on consumer hardware, with a temporal 80/10/10 train/val/test split. After filtering to users with at least 5 interactions, ~110k unique users remain.

---

## Results

Evaluated on the **active-item pool** (10,556 of 20,000 catalogue items that appear in at least one transaction), which reflects realistic retrieval difficulty.

| Model | Val R@10 | Val NDCG@10 | Test R@10 | Test NDCG@10 |
|-------|----------|-------------|-----------|--------------|
| Item popularity baseline | 0.0227 | 0.0107 | 0.0107 | 0.0048 |
| Text-only two-tower | 0.0298 | 0.0184 | 0.0248 | 0.0151 |
| **Multimodal two-tower (ours)** | **0.0445** | **0.0285** | **0.0328** | **0.0208** |

- Multimodal vs text-only: **+49% relative** on val Recall@10, confirming image signal carries genuine style information beyond the text description.
- Active-item pool vs full 20k catalogue: **+10% relative** Recall@10, expected since restricting to items users actually interacted with removes ~9,500 effectively-dead catalogue entries.

### Training details

| Hyperparameter | Value |
|---|---|
| Epochs | 16 (early stopping, patience=2 on val R@10) |
| Best epoch | 11 |
| Batch size | 256 |
| Optimiser | AdamW |
| Learning rate | 3×10⁻⁴ |
| Weight decay | 1×10⁻⁵ |
| Temperature (τ) | 0.1 |
| LR warmup steps | 500 |
| Mixed precision | FP16 (torch.amp) |
| Hardware | NVIDIA RTX 3070 Laptop GPU (8GB) |

---

## Demo

*Screenshots coming after HuggingFace Spaces deployment.*

The demo lets you select a test-set user, view their 5 most recent browsing items, and retrieve top-K recommendations — each with a one-sentence LLM-generated explanation grounded in the user's visible style patterns.

---

## Tech stack

| Component | Library |
|---|---|
| Image encoder | CLIP ViT-B/32 via `open_clip` |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Training | PyTorch 2.5 + `torch.amp` (FP16) |
| Retrieval index | FAISS `IndexFlatIP` |
| LLM explanations (local) | Ollama + LLaMA 3.1 8B |
| LLM explanations (cloud) | Groq API + LLaMA 3.1 8B |
| Demo UI | Streamlit |
| Deployment | HuggingFace Spaces |

---

## Running locally

```bash
git clone https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender
cd multimodal-fashion-recommender
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# download H&M data from Kaggle to data/h-and-m-personalized-fashion-recommendations/
# (articles.csv, transactions_train.csv, images/)

python scripts/00_prepare_data.py
python scripts/01_build_embeddings.py
python scripts/02_train_model.py
python scripts/03_build_index.py

# start Ollama with LLaMA 3.1
ollama pull llama3.1:8b

# launch demo
streamlit run app/streamlit_app.py
```

---

## Project structure

```
multimodal-fashion-recommender/
├── config.yaml                          # all hyperparameters
├── requirements.txt
├── app/
│   └── streamlit_app.py                 # demo UI
├── src/
│   ├── data/
│   │   ├── loader.py                    # H&M CSV loaders
│   │   └── preprocess.py               # temporal split, user sequences
│   ├── encoders/
│   │   ├── image_encoder.py             # CLIP wrapper
│   │   └── text_encoder.py             # SBERT wrapper
│   ├── models/
│   │   ├── item_tower.py               # MLP fusion (image + text -> 256d)
│   │   ├── user_tower.py               # Transformer sequence encoder
│   │   └── two_tower.py                # combined model, in-batch InfoNCE
│   ├── training/
│   │   ├── dataset.py                  # FashionInteractionDataset
│   │   ├── train.py                    # training loop, early stopping
│   │   └── evaluate.py                 # Recall@K, NDCG@K
│   ├── retrieval/
│   │   └── faiss_index.py              # FaissRetriever (IndexFlatIP)
│   └── reasoning/
│       └── llm_explainer.py            # OllamaExplainer / GroqExplainer
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_build_embeddings.py
│   ├── 02_train_model.py
│   ├── 03_build_index.py
│   ├── 04_baselines.py
│   └── 05_eval_comparison.py
└── spaces/                              # HuggingFace Spaces deployment
    ├── app.py
    ├── requirements.txt
    └── config_spaces.yaml
```

---

## What I learned

- **Representation collapse is sneaky.** Early training runs produced near-zero Recall@10 (< 0.001). The cause was a combination of learning rate too high (1e-3), no warmup, and temperature too low (0.07) — all three together pushed item embeddings to collapse onto a single point. The fix was dropping LR to 3e-4, adding 500-step warmup, and raising temperature to 0.1. Monitoring mean pairwise cosine similarity among random item batches every 100 steps was the key diagnostic.

- **Eval-time history leakage is subtle.** Initial val/test metrics were inflated because user history was built only from the training split. When a val-set user's most recent purchase was in val, it was excluded from their history — making the sequence shorter and noisier, but not causing leakage. The real bug was the *opposite*: the training history didn't include val-period purchases, so the user representation at eval time was stale. Fixing history to include all transactions up to the target gave a consistent +3-4% relative improvement.

- **Image signal genuinely helps, but only after text is working.** The text-only baseline (val R@10 = 0.0298) was non-trivial — SBERT embeddings for H&M product descriptions carry strong style signal. Adding CLIP image embeddings lifted val R@10 to 0.0445 (+49% relative). The gain would have been invisible had the model not already learned to use text well; multimodal training is not free — it only pays off once the unimodal signal is well-exploited.

- **Active-item filtering is worth implementing.** Restricting the retrieval pool from 20k to 10,556 active items improved Recall@10 by ~10% relative. Dead catalogue items (no transactions in any split) act as distractors during retrieval. In production, a freshness-weighted active set would be standard practice.

- **Full retrieval eval beats train-loss early stopping.** Early versions used val loss for early stopping. Switching to val Recall@10 as the early-stopping criterion found a strictly better checkpoint (epoch 11 vs epoch 8 under loss stopping), because the contrastive loss and retrieval rank don't move in lockstep — a small drop in loss can precede a large jump in top-K recall.

---

## License

MIT
