# PLAN.md — Multimodal Fashion Recommender with LLM-Generated Explanations

> **Claude Code: read this entire file before writing any code. Follow phases in order. After each phase, run the verification commands and commit to git. Do not skip phases or combine them.**

---

## Project overview

A two-tower recommendation system for fashion products that combines CLIP image embeddings with sentence-transformer text embeddings over item metadata, paired with a session-aware sequence model that captures user intent from recent browsing history. After retrieval, a local LLM (via Ollama) generates a one-sentence natural-language explanation for each recommendation.

**Target user:** a fashion e-commerce customer browsing a catalogue. Given a user's recent interaction history (items viewed/purchased), the system retrieves the top-K similar items and explains *why* each is recommended.

**Key design decisions (do not change without asking):**
- Dataset: H&M Personalized Fashion Recommendations (Kaggle), already downloaded to `data/`
- Image encoder: CLIP ViT-B/32 (frozen, via `open_clip` or `transformers`)
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2` (frozen)
- Fusion: late fusion — concatenate 512-dim image + 384-dim text → project to 256-dim via MLP
- User tower: mean-pool user's last 20 item embeddings → project via MLP to 256-dim
- Training loss: in-batch contrastive (InfoNCE) with temperature 0.07
- Retrieval: FAISS flat index for top-K cosine similarity
- Explanation LLM: `llama3.1:8b` via Ollama REST API at `http://localhost:11434`
- Demo UI: Streamlit
- Deployment: HuggingFace Spaces (Streamlit SDK)

---

## Environment setup (do this FIRST, before Phase 1)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install --upgrade pip setuptools wheel
```

Create `requirements.txt` with these exact versions (tested against Python 3.13; if any fail to install, fall back to Python 3.11):

```
torch>=2.4.0
torchvision>=0.19.0
transformers>=4.44.0
sentence-transformers>=3.0.0
open-clip-torch>=2.24.0
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=15.0.0
pillow>=10.0.0
faiss-cpu>=1.8.0
scikit-learn>=1.4.0
tqdm>=4.66.0
streamlit>=1.38.0
requests>=2.32.0
pyyaml>=6.0.0
matplotlib>=3.8.0
ipykernel>=6.29.0
```

**If `faiss-cpu` fails on Python 3.13:** install Python 3.11 alongside, recreate venv with `py -3.11 -m venv .venv`, and try again. faiss often lags on latest Python.

**Install PyTorch with CUDA for RTX 3070:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU is available:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
Should print `True NVIDIA GeForce RTX 3070 Laptop GPU`.

---

## Repository structure (final state)

```
multimodal-fashion-recommender/
├── README.md
├── PLAN.md                          (this file)
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/                            (not committed — see .gitignore)
│   ├── raw/                         (H&M CSVs + images)
│   └── processed/                   (embeddings, splits)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                (Phase 1)
│   │   └── preprocess.py            (Phase 1)
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── image_encoder.py         (Phase 2)
│   │   └── text_encoder.py          (Phase 2)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── item_tower.py            (Phase 3)
│   │   ├── user_tower.py            (Phase 3)
│   │   └── two_tower.py             (Phase 3)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py               (Phase 3)
│   │   ├── train.py                 (Phase 4)
│   │   └── evaluate.py              (Phase 4)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── faiss_index.py           (Phase 4)
│   └── reasoning/
│       ├── __init__.py
│       └── llm_explainer.py         (Phase 5)
├── app/
│   └── streamlit_app.py             (Phase 5)
├── scripts/
│   ├── 01_build_embeddings.py       (Phase 2)
│   ├── 02_train_model.py            (Phase 4)
│   └── 03_build_index.py            (Phase 4)
├── notebooks/
│   └── 01_data_exploration.ipynb    (Phase 1)
├── tests/
│   ├── test_encoders.py
│   └── test_models.py
└── checkpoints/                     (not committed)
```

---

## `.gitignore` (create this in Phase 0)

```
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
checkpoints/
*.ckpt
.ipynb_checkpoints/
.DS_Store
*.log
.env
```

---

## `config.yaml` (create this in Phase 0)

```yaml
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  images_dir: "data/raw/images"
  articles_csv: "data/raw/articles.csv"
  transactions_csv: "data/raw/transactions_train.csv"
  customers_csv: "data/raw/customers.csv"
  # For development, subsample so we don't process 105k items and 31M transactions
  sample_num_items: 20000
  sample_num_transactions: 500000
  min_interactions_per_user: 5

encoders:
  image_model: "ViT-B-32"
  image_pretrained: "openai"
  image_embed_dim: 512
  text_model: "sentence-transformers/all-MiniLM-L6-v2"
  text_embed_dim: 384
  device: "cuda"
  batch_size: 64

model:
  item_fusion_hidden: 512
  output_dim: 256
  user_seq_len: 20
  dropout: 0.2

training:
  batch_size: 256
  num_epochs: 5
  lr: 0.001
  weight_decay: 0.00001
  temperature: 0.07
  num_negatives: "in_batch"
  val_split: 0.1
  test_split: 0.1
  seed: 42

retrieval:
  index_type: "flat"
  top_k: 10
  metric: "cosine"

llm:
  provider: "ollama"
  model: "llama3.1:8b"
  host: "http://localhost:11434"
  temperature: 0.3
  max_tokens: 80
```

---

## PHASE 0 — Scaffolding (15 minutes)

**Goal:** Empty skeleton with venv, requirements installed, config, .gitignore, and initial README stub.

**Tasks:**
1. Create venv, activate, install requirements (see Environment setup above)
2. Create all empty `__init__.py` files in `src/` subdirectories
3. Create `config.yaml`, `.gitignore`, and a stub `README.md` (just the title + one-line description for now; full README comes in Phase 5)
4. Initialize git, make first commit

**Verification:**
```bash
python -c "import torch, transformers, open_clip, sentence_transformers, faiss, streamlit; print('All imports OK')"
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"
```

**Commit message:** `Phase 0: scaffolding, venv, requirements, config`

---

## PHASE 1 — Data pipeline (~2 hours)

**Goal:** Load, clean, and split H&M data. Produce working data loaders for items, users, and interactions.

### Files to create

**`src/data/loader.py`** — pure data loading, no transformations

Functions required:
- `load_articles(config) -> pd.DataFrame`: loads articles.csv, keeps columns: `article_id`, `prod_name`, `product_type_name`, `product_group_name`, `colour_group_name`, `department_name`, `detail_desc`. Drop rows where `detail_desc` is NaN. If `sample_num_items` is set, random-sample that many (seed from config).
- `load_transactions(config) -> pd.DataFrame`: loads transactions_train.csv, keeps `t_dat`, `customer_id`, `article_id`. Filter to only include articles present in the sampled article set. If `sample_num_transactions` is set, take the most recent N.
- `load_customers(config) -> pd.DataFrame`: loads customers.csv if present, keeps `customer_id` only (we don't need demographics).
- `get_image_path(article_id, config) -> Path`: returns the expected image path. H&M uses `images/{first_3_chars}/{article_id}.jpg` structure.

**`src/data/preprocess.py`** — transformations and split logic

Functions required:
- `build_item_text(articles_df) -> pd.DataFrame`: adds a `full_text` column concatenating `prod_name`, `product_type_name`, `colour_group_name`, and `detail_desc`, separated by ". " — this is what the text encoder will embed. Example: `"Black slim-fit jeans. Trousers. Black. Five-pocket jeans in washed denim with a zip fly."`
- `filter_cold_users(transactions_df, min_interactions) -> pd.DataFrame`: drop users with fewer than `min_interactions_per_user` transactions.
- `temporal_split(transactions_df, val_frac, test_frac) -> (train, val, test)`: sort by `t_dat`, take last `test_frac` as test, next-to-last `val_frac` as val, remainder as train. Temporal split, NOT random — this is important for recommender evaluation.
- `build_user_sequences(train_df, seq_len) -> dict[user_id, list[article_id]]`: for each user, their last `seq_len` items in chronological order.
- `save_processed(objects: dict, config)`: saves DataFrames as parquet and dicts as pickle to `data/processed/`.

**`notebooks/01_data_exploration.ipynb`** — quick EDA
- Show number of unique items, users, interactions after filtering
- Distribution of product types and colours
- Display a few sample images with captions
- Interaction count histogram
- Date range covered

### Verification (Claude Code: run this after writing the above)

Create `scripts/00_prepare_data.py` that runs the full pipeline end-to-end and prints a summary:

```python
# scripts/00_prepare_data.py
import yaml
from src.data.loader import load_articles, load_transactions
from src.data.preprocess import (
    build_item_text, filter_cold_users, temporal_split,
    build_user_sequences, save_processed
)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

articles = load_articles(cfg)
articles = build_item_text(articles)
transactions = load_transactions(cfg)
transactions = filter_cold_users(transactions, cfg["data"]["min_interactions_per_user"])
train, val, test = temporal_split(transactions, cfg["training"]["val_split"], cfg["training"]["test_split"])
user_seqs = build_user_sequences(train, cfg["model"]["user_seq_len"])

print(f"Articles: {len(articles):,}")
print(f"Train/Val/Test transactions: {len(train):,} / {len(val):,} / {len(test):,}")
print(f"Unique users with sequences: {len(user_seqs):,}")
print(f"Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")

save_processed({
    "articles": articles,
    "train": train, "val": val, "test": test,
    "user_seqs": user_seqs,
}, cfg)
print("Saved to data/processed/")
```

Run it. Expected output shape: ~20k articles, ~400k/50k/50k train/val/test, tens of thousands of users. If numbers look way off, stop and debug before proceeding.

**Commit message:** `Phase 1: H&M data pipeline with temporal splits and user sequences`

---

## PHASE 2 — Encoders + embedding generation (~2 hours, GPU-heavy)

**Goal:** Precompute fixed image + text embeddings for all items. These embeddings are cached to disk so training doesn't re-run them every epoch.

### Files to create

**`src/encoders/image_encoder.py`**

```python
class ImageEncoder:
    """Wraps CLIP ViT-B/32. Frozen — no training. Returns 512-dim normalised embeddings."""

    def __init__(self, config): ...
    def encode_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Returns (N, 512) array. Handles missing/corrupt images by returning zero vectors."""
    def encode_directory(self, articles_df, images_dir, output_path, batch_size=64):
        """Iterates all articles, encodes their images, saves to .npy. Prints tqdm progress.
        Handles missing images gracefully — log warnings, use zero vectors."""
```

Implementation notes for Claude Code:
- Use `open_clip` library for CLIP (more reliable than HF transformers for CLIP).
- Load model with `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")`.
- Move to CUDA, set `.eval()`, wrap all forward passes in `torch.no_grad()`.
- Normalise output embeddings with `embeddings / embeddings.norm(dim=-1, keepdim=True)`.
- Save output as `data/processed/item_image_embeddings.npy` with a parallel `data/processed/item_ids_image.npy` tracking ordering.

**`src/encoders/text_encoder.py`**

```python
class TextEncoder:
    """Wraps sentence-transformers. Frozen. Returns 384-dim normalised embeddings."""

    def __init__(self, config): ...
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Returns (N, 384) array."""
    def encode_dataframe(self, articles_df, text_col, output_path, batch_size=128):
        """Encodes articles_df[text_col], saves to .npy."""
```

Implementation notes:
- Use `SentenceTransformer(model_name, device="cuda")`.
- The library's `encode(texts, normalize_embeddings=True)` does normalisation for you.
- Save as `data/processed/item_text_embeddings.npy`.

**`scripts/01_build_embeddings.py`** — orchestrator

Loads articles, instantiates both encoders, generates both embedding matrices, saves to disk, and prints timing + memory usage.

### Verification

After running `python scripts/01_build_embeddings.py`:
- `data/processed/item_image_embeddings.npy` exists, shape `(~20000, 512)`
- `data/processed/item_text_embeddings.npy` exists, shape `(~20000, 384)`
- Both have L2-normalised rows (check a sample: `np.linalg.norm(x, axis=1).mean()` ≈ 1.0)

Write `tests/test_encoders.py` with at least:
- Test that encoding a single known image returns a 512-dim vector with norm 1.0 (±1e-5)
- Test that encoding a text string returns a 384-dim vector with norm 1.0 (±1e-5)
- Test that encoding the same input twice gives identical outputs (determinism)

**Commit message:** `Phase 2: CLIP image + SBERT text encoders with precomputed embeddings`

---

## PHASE 3 — Two-tower model (~1.5 hours, mostly design work)

**Goal:** Define the model architecture and training dataset class. No training yet.

### Files to create

**`src/models/item_tower.py`**

```python
class ItemTower(nn.Module):
    """Fuses precomputed image + text embeddings into a unified 256-dim item vector."""
    def __init__(self, image_dim=512, text_dim=384, hidden=512, output_dim=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, image_emb, text_emb):
        # image_emb: (B, 512), text_emb: (B, 384)
        x = torch.cat([image_emb, text_emb], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)  # L2-normalise output
```

**`src/models/user_tower.py`**

```python
class UserTower(nn.Module):
    """
    Sequence-aware user representation.
    Input: user's last-N item vectors from ItemTower (already 256-dim).
    Output: single 256-dim user vector.

    Uses a shallow transformer (2 layers) with mean-pooling over sequence.
    """
    def __init__(self, item_dim=256, n_heads=4, n_layers=2, dropout=0.2, max_seq=20):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq, item_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=item_dim, nhead=n_heads, dim_feedforward=item_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, item_seq_emb, attention_mask=None):
        # item_seq_emb: (B, N, 256)
        B, N, D = item_seq_emb.shape
        pos = self.pos_emb(torch.arange(N, device=item_seq_emb.device))
        x = item_seq_emb + pos
        # Mask padded positions; attention_mask is (B, N), True = real token
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask  # transformer wants True = pad
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # Masked mean pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return F.normalize(x, dim=-1)
```

**`src/models/two_tower.py`**

```python
class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.item_tower = ItemTower(
            image_dim=config["encoders"]["image_embed_dim"],
            text_dim=config["encoders"]["text_embed_dim"],
            hidden=config["model"]["item_fusion_hidden"],
            output_dim=config["model"]["output_dim"],
            dropout=config["model"]["dropout"],
        )
        self.user_tower = UserTower(
            item_dim=config["model"]["output_dim"],
            max_seq=config["model"]["user_seq_len"],
            dropout=config["model"]["dropout"],
        )
        self.temperature = config["training"]["temperature"]

    def forward(self, user_seq_img, user_seq_txt, user_mask,
                target_img, target_txt):
        # Encode user's history items through item tower
        B, N, _ = user_seq_img.shape
        seq_img_flat = user_seq_img.view(B * N, -1)
        seq_txt_flat = user_seq_txt.view(B * N, -1)
        seq_item = self.item_tower(seq_img_flat, seq_txt_flat).view(B, N, -1)
        user_emb = self.user_tower(seq_item, user_mask)
        target_emb = self.item_tower(target_img, target_txt)
        # Similarity matrix: (B, B), diagonal = positive pairs, off-diagonal = in-batch negatives
        logits = user_emb @ target_emb.T / self.temperature
        return logits
```

**`src/training/dataset.py`**

```python
class FashionInteractionDataset(Dataset):
    """
    Each sample: one (user, target_item) positive pair.
    Returns: user's history sequence embeddings + target item embeddings.

    History is the user's last N items STRICTLY BEFORE the target timestamp
    (no leakage). Pad shorter sequences with zeros and track with attention mask.
    """
    def __init__(self, interactions_df, user_seqs, item_img_emb, item_txt_emb,
                 article_id_to_idx, seq_len=20):
        ...
    def __len__(self): ...
    def __getitem__(self, idx):
        # Returns a dict with:
        #   user_seq_img: (seq_len, 512)
        #   user_seq_txt: (seq_len, 384)
        #   user_mask:    (seq_len,) bool
        #   target_img:   (512,)
        #   target_txt:   (384,)
        ...
```

Key implementation detail: the dataset needs access to the `item_ids_image.npy` ordering to look up embeddings by `article_id`. Build a dict `article_id -> row_index` at init.

### Verification

Write `tests/test_models.py`:
- Instantiate `TwoTowerModel` with dummy config
- Pass a random batch through it (batch_size=4, seq_len=20)
- Assert output logits shape is `(4, 4)` (in-batch similarity)
- Assert item tower output is unit-normalised

**Commit message:** `Phase 3: two-tower model with item fusion and user sequence tower`

---

## PHASE 4 — Training + retrieval (~3 hours, GPU-heavy)

**Goal:** Train the model, evaluate on val/test with Recall@10 and NDCG@10, build FAISS index for inference.

### Files to create

**`src/training/train.py`** — training loop

Requirements:
- Use `torch.optim.AdamW` with `lr` and `weight_decay` from config
- Use `torch.nn.functional.cross_entropy` with targets = `torch.arange(batch_size)` (for in-batch contrastive — diagonal is positive)
- Track train loss, val loss per epoch
- Early stopping on val loss (patience=2)
- Save best checkpoint to `checkpoints/best.pt`
- Log every 50 steps with tqdm
- Use mixed precision (`torch.cuda.amp.autocast`) — RTX 3070 benefits significantly
- At end of each epoch, compute Recall@10 on val set (see below)

**`src/training/evaluate.py`** — metrics

```python
def recall_at_k(user_embs, item_embs, true_item_indices, k=10) -> float:
    """
    user_embs: (N, D)
    item_embs: (M, D) — all catalogue items
    true_item_indices: (N,) — index of true next item for each user
    Returns: fraction of users for which the true item is in top-K.
    """

def ndcg_at_k(user_embs, item_embs, true_item_indices, k=10) -> float:
    """Normalised discounted cumulative gain."""
```

Implementation: batch the user→item similarity computation (don't materialise full `N x M` matrix if it's huge; chunk users).

**`src/retrieval/faiss_index.py`**

```python
class FaissRetriever:
    def __init__(self, item_embs: np.ndarray, article_ids: list, metric="cosine"):
        # For cosine similarity with normalised vectors, use IndexFlatIP (inner product)
        ...
    def search(self, query_emb: np.ndarray, k=10) -> list[tuple[str, float]]:
        """Returns list of (article_id, score) pairs."""
    def save(self, path): ...
    @classmethod
    def load(cls, path, articles_df): ...
```

**`scripts/02_train_model.py`** — orchestrator that calls training loop

**`scripts/03_build_index.py`** — after training, encode all items through ItemTower and build FAISS index, save to disk.

### Verification

After running training (5 epochs should take ~20-30 minutes on your RTX 3070 with the 20k-item sample):
- Train loss should decrease steadily
- Val Recall@10 should be **>0.05 at minimum** — if it's below 0.02, something is wrong (most likely data leakage in sequence building, or mismatched embeddings)
- Sanity check the index: pick any known item, search with its own embedding as query, verify it's in the top-3 (usually top-1)

If metrics are bad, the most common bugs:
1. Target item is included in user's history sequence (leakage) — check your dataset class carefully
2. Image/text embeddings not aligned with article_id ordering (off-by-one, wrong index)
3. Temperature too high or too low — try 0.05 or 0.1

**Commit message:** `Phase 4: training loop with in-batch contrastive loss, evaluation metrics, FAISS index`

---

## PHASE 5 — LLM explanation layer + Streamlit app (~2 hours)

**Goal:** After retrieval, generate a natural-language explanation for why each recommended item fits the user's browsing pattern. Wrap everything in a demo UI.

### Files to create

**`src/reasoning/llm_explainer.py`**

```python
class OllamaExplainer:
    def __init__(self, config):
        self.host = config["llm"]["host"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]

    def explain(self, user_history: list[dict], recommended_item: dict) -> str:
        """
        user_history: list of dicts with keys 'prod_name', 'product_type_name', 'colour_group_name'
        recommended_item: same schema
        Returns: one-sentence natural-language explanation.

        Uses ollama /api/generate endpoint with streaming=False for simplicity.
        """
        prompt = self._build_prompt(user_history, recommended_item)
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _build_prompt(self, user_history, rec_item):
        history_str = "\n".join(
            f"- {h['prod_name']} ({h['colour_group_name']} {h['product_type_name']})"
            for h in user_history[-5:]
        )
        return f"""You are a concise fashion assistant. A user recently browsed these items:

{history_str}

We are recommending: {rec_item['prod_name']} ({rec_item['colour_group_name']} {rec_item['product_type_name']})

In ONE sentence (max 25 words), explain why this recommendation fits the user's style. Be specific about patterns (e.g., colour, product category, style). Do not use bullet points. Do not start with 'This' or 'The'."""
```

**`app/streamlit_app.py`**

Requirements:
- Sidebar: user selector (random sample of users from the test set, or free-text article_id entry)
- Main area:
  - Left column: user's recent history (show last 5 items with thumbnails, product name, colour)
  - Right column: top-5 recommendations with thumbnails, and under each, the LLM-generated explanation
- Load model, FAISS index, and articles DataFrame once via `@st.cache_resource`
- Button to regenerate recommendations (useful for demoing)
- Loading spinner while LLM generates explanations (takes 2-4 seconds per item with 8B local model — show progress)
- Small footer: "Built with CLIP + BERT + Llama 3.1 via Ollama. [GitHub link]"

Visual layout hint: use `st.image` with `width=150` for thumbnails, `st.caption` for metadata, and `st.info` for the explanation text.

### Verification

Run:
```bash
# Terminal 1: make sure ollama is running
ollama serve   # usually already running as a service on Windows

# Terminal 2: run the demo
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, select a user, click "Recommend", confirm:
- Top-5 items appear with thumbnails
- Each has an explanation that reads naturally and references the user's history
- No crashes or timeouts

**Commit message:** `Phase 5: Ollama explanation layer and Streamlit demo UI`

---

## PHASE 6 — README + deployment (~1.5 hours)

**Goal:** A README that looks recruiter-ready, plus a HuggingFace Spaces deployment.

### `README.md` content

Follow this exact structure:

````markdown
# Multimodal Fashion Recommender with LLM-Generated Explanations

> A two-tower recommender combining CLIP image embeddings, sentence-transformer text encoders, and session-aware sequence modelling — with an LLM reasoning layer that generates natural-language explanations for each recommendation.

🔗 **[Live Demo on HuggingFace Spaces](<link>)** · 🎥 **[Demo Video / Screenshots](#demo)**

---

## Motivation

Modern recommender systems are accurate but opaque. Users and product teams both benefit from knowing *why* an item is recommended. This project explores a practical way to combine retrieval-quality two-tower models with LLM-based post-hoc explanations, specifically for fashion e-commerce.

## Architecture

<ASCII or image diagram showing:
  (Catalogue images → CLIP) ┐
                             ├→ Item Tower (MLP) → 256-dim
  (Catalogue text  → SBERT) ┘
  
  (User's last 20 items → Item Tower → Transformer encoder → mean-pool) → User Tower → 256-dim
  
  User ⨯ Item → cosine similarity → top-K → LLM reasoning layer → natural-language explanation
>

### Key design decisions
- **Two-tower architecture** for scalable retrieval (user and items encoded independently; FAISS index for millions of items)
- **Frozen encoders** (CLIP, SBERT) keep training cheap and avoid catastrophic forgetting of visual/textual knowledge
- **In-batch contrastive loss** (InfoNCE, τ=0.07) — no explicit negative sampling needed
- **Temporal train/val/test split** — prevents leakage that plagues many recommender benchmarks
- **Post-retrieval LLM explanation** — keeps latency low (LLM called only K times, not ranking millions)

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 105k articles, 31M transactions, 1.4M customers. This project uses a 20k-article sample for training tractability on consumer hardware.

## Results

| Model                          | Recall@10 | NDCG@10 |
|--------------------------------|-----------|---------|
| Item popularity baseline       | 0.0XX     | 0.0XX   |
| Text-only two-tower            | 0.0XX     | 0.0XX   |
| **Multimodal two-tower (ours)**| **0.0XX** | **0.0XX**|

*(fill in actual numbers after Phase 4)*

## Demo

<screenshots go here>

## Tech stack

PyTorch · CLIP (open_clip) · Sentence-Transformers · FAISS · Ollama · Streamlit · HuggingFace Hub

## Running locally

```bash
git clone https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender
cd multimodal-fashion-recommender
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# download H&M data to data/raw/ (see data/README.md)
python scripts/00_prepare_data.py
python scripts/01_build_embeddings.py
python scripts/02_train_model.py
python scripts/03_build_index.py

# start ollama with a model for explanations
ollama pull llama3.1:8b
ollama serve

# run demo
streamlit run app/streamlit_app.py
```

## Project structure

<file tree from repo structure section>

## What I learned

<write 3-5 honest bullets here after you finish — things like "late fusion outperformed early fusion by X% in my experiments", "LLM explanations dramatically improved qualitative evaluation but needed careful prompt engineering to avoid hallucinations about product attributes", etc.>

## License

MIT
````

### HuggingFace Spaces deployment

1. Create account at https://huggingface.co
2. Create a new Space: https://huggingface.co/new-space
   - Name: `multimodal-fashion-recommender`
   - SDK: Streamlit
   - Hardware: CPU basic (free tier)
3. Because the free tier has no GPU and limited RAM, **the Space version uses a smaller catalogue (5k items), precomputed embeddings checked in as LFS artifacts, and calls an LLM API instead of Ollama**. Create a branch `spaces-deploy` with these modifications:
   - Replace `OllamaExplainer` with `GroqExplainer` that calls Groq's free-tier API (Llama 3.1 8B, fast)
   - Reduce `sample_num_items` to 5000 in a `config_spaces.yaml`
   - Add `huggingface_hub` to requirements, upload embeddings + FAISS index as release artifacts
   - App loads these at startup
4. Configure Space secrets: add `GROQ_API_KEY` via Space settings.
5. Push to the Space's git remote.

The Ollama version is what runs locally and is what you show in videos. The Spaces version is what recruiters can click on instantly.

**Commit message:** `Phase 6: README with architecture, results, and HuggingFace Spaces deployment`

---

## Final acceptance criteria

Before marking this project done, verify:

- [ ] Repo is public at `github.com/gaurav-gandhi-2411/multimodal-fashion-recommender`
- [ ] `README.md` has a working live demo link
- [ ] Results table has real numbers (not placeholders)
- [ ] Running `streamlit run app/streamlit_app.py` locally produces recommendations with explanations
- [ ] All tests in `tests/` pass via `pytest`
- [ ] The "What I learned" section is written honestly and reflects actual findings
- [ ] At least one screenshot or GIF in README showing the demo working
- [ ] HuggingFace Space loads and produces working recommendations within 30 seconds

---

## Claude Code: guidelines when implementing this

1. **Stop after each phase and summarise what you did + what you're about to do next.** I'll review before you proceed.
2. **Run verification commands after each phase and paste the output.** Don't mark a phase complete based on "it looked right."
3. **If a dependency fails to install, tell me immediately — don't work around it silently.** Especially for `faiss-cpu` on Python 3.13.
4. **Keep commits atomic and phase-aligned.** One commit per phase, with the exact commit message given above.
5. **Do not modify `config.yaml` values during implementation.** If you think a value is wrong, tell me and we'll discuss.
6. **Handle errors gracefully but loudly.** Missing images, NaN in data, Ollama not running — log a clear warning and fall back sensibly, don't silently produce garbage.
7. **Prefer simplicity over cleverness.** A 30-line function that works beats a 10-line function that's hard to debug.
