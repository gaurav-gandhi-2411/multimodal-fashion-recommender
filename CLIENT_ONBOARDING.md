# Client Onboarding Guide

Get from "here's our catalog" to live recommendations in under 30 minutes.

## Prerequisites

- Python 3.11+ installed in your environment
- API credentials provided by your integration contact (you will receive a `<BRAND>_API_KEY` value)
- One of the following catalog sources:
  - A Shopify store with `/products.json` enabled (most stores have this by default)
  - A CSV export of your product catalog (column requirements below)

---

## Step 1 — Clone the repository and install dependencies

```bash
git clone https://github.com/<org>/multimodal-fashion-recommender.git
cd multimodal-fashion-recommender
pip install uv
uv sync --extra ml
```

**Expected result:** Dependencies install without errors. The `[ml]` extra pulls in CLIP and sentence-transformers — required for embedding your catalog.

---

## Step 2 — Prepare your catalog

### Option A — Shopify (recommended)

No export needed. Just provide your store URL. The ingestion script paginates `/products.json` automatically (250 products per page).

```bash
python scripts/ingest_catalog.py \
  --source shopify \
  --url https://yourstore.myshopify.com/products.json \
  --brand your_brand
```

If your store has `/products.json` disabled, proceed to Option B.

### Option B — CSV file

Export a CSV with these exact column names (additional columns are ignored):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `product_id` | string | Unique product identifier | `SKU-001` |
| `title` | string | Product name | `Oversized Drop-Shoulder Tee` |
| `description` | string | Product description (HTML stripped automatically) | `100% cotton, relaxed fit` |
| `image_url` | string | Full URL to the primary product image | `https://cdn.example.com/sku001.jpg` |
| `price_inr` | number | Price in INR (must be > 0) | `1299` |
| `category` | string | Product category | `topwear` |
| `pdp_url` | string | Full URL to the product page | `https://yourstore.com/products/sku-001` |

```bash
python scripts/ingest_catalog.py \
  --source csv \
  --path /path/to/your/catalog.csv \
  --brand your_brand
```

**Expected output:**
```
INFO Step 1/6 — downloading images for 150 items
INFO Step 2/6 — CLIP image encoding (device=cuda)
INFO Step 3/6 — SBERT text encoding
INFO Step 4/6 — ItemTower fusion
INFO Step 5/6 — building FAISS index
INFO Step 6/6 — writing catalog parquet
INFO Brand YAML written → brands/your_brand.yaml

Next steps:
  1. Set the API key:  export YOUR_BRAND_API_KEY=<your-key>
  2. Start the server: uvicorn app.api.main:app --reload
  3. Test cold-start:  curl ...
```

**Files written:**
```
data/your_brand/images/          ← downloaded product images
data/your_brand/items.parquet    ← catalog with article_ids
indices/your_brand/active.faiss/ ← FAISS index (queryable immediately)
indices/your_brand/item_emb.npy  ← 256-dim item embeddings
brands/your_brand.yaml           ← brand config (loaded at API startup)
```

**Resumability:** If the script is interrupted during image download, re-run the same command. Already-downloaded images are skipped. Failed images are recorded in `data/your_brand/images/failed_images.json`.

---

## Step 3 — (Optional) Ingest interaction history

Without interaction data, the API serves item-to-item similarity (cold-start mode). This already powers "similar items" and "complete the look" use cases.

To enable **personalized recommendations** (`/v1/{brand}/recommend` with a `user_id`), provide your purchase or event history.

### Option A — Shopify orders export

Export your orders from the Shopify admin: **Orders → Export → All orders → CSV for Excel**.

```bash
python scripts/ingest_interactions.py \
  --brand your_brand \
  --path /path/to/shopify_orders.csv \
  --format shopify
```

The script reads `Email`, `Paid at`, and `Lineitem sku` columns from the Shopify export.

### Option B — Generic events CSV

Any CSV with these columns works:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Customer identifier |
| `product_id` | string | Must match `product_id` in your catalog |
| `timestamp` | ISO 8601 | e.g. `2025-11-01T10:00:00Z` |
| `event_type` | string | One of: `purchase`, `view`, `wishlist`, `cart` |

```bash
python scripts/ingest_interactions.py \
  --brand your_brand \
  --path /path/to/events.csv
```

**Expected output:**
```
INFO Loaded catalog: 150 products
INFO Loaded 4200 raw interaction rows
INFO 4150 rows after mapping to article_ids
INFO Split: train=3320  val=415  test=415
INFO Wrote 3320 rows → data/your_brand/transactions/train.parquet
INFO Wrote 415 rows → data/your_brand/transactions/val.parquet
INFO Wrote 415 rows → data/your_brand/transactions/test.parquet
INFO Updated brands/your_brand.yaml with transactions_dir
```

---

## Step 4 — Set the API key and start the server

```bash
export YOUR_BRAND_API_KEY=<key-provided-by-integration-contact>
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Verify the brand loaded correctly:

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "ok",
  "brands": [
    {
      "brand": "your_brand",
      "display_name": "Your Brand",
      "item_count": 150
    }
  ]
}
```

---

## Step 5 — Test your first recommendation call

### Item-to-item similarity (cold-start, no interaction history needed)

```bash
curl -H "X-Api-Key: <your-key>" \
  http://localhost:8000/v1/your_brand/item/1/similar?k=5
```

This returns the 5 most visually and semantically similar items to item `1`.

### Personalized recommendations (requires Step 3)

```bash
curl -X POST \
  -H "X-Api-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice@example.com", "k": 10}' \
  http://localhost:8000/v1/your_brand/recommend
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `RuntimeError: /products.json returned 404` | Store has JSON API disabled | Use `--source csv` instead |
| `ValueError: Generic CSV missing required columns: ['price_inr']` | Missing column in your CSV | Add the column (see column table in Step 2B) |
| `FileNotFoundError: Catalog parquet not found` | Ran `ingest_interactions` before `ingest_catalog` | Run Step 2 first |
| `50%+ of catalog rows invalid` | Widespread data quality issue | Check CSV encoding; ensure `price_inr` > 0 for all rows |
| Images downloading slowly | Large catalog with many images | Add `--workers 16` to `ingest_catalog.py` (default: 8) |
| `ModuleNotFoundError: No module named 'open_clip'` | Installed without `[ml]` extra | Re-install: `uv sync --extra ml` |

## Notes on robots.txt

By default, the Shopify ingestion logs a warning if robots.txt restricts `/products.json` but proceeds anyway — you are ingesting **your own store** under an authorized onboarding agreement. To enforce strict robots.txt compliance (e.g. for third-party stores), pass `--respect-robots`.

---

## What happens next

Once the brand is live:
- `/v1/your_brand/item/{id}/similar` is immediately available (cold-start, no interaction data needed)
- `/v1/your_brand/recommend` with `user_id` requires interaction data from Step 3
- Re-running `ingest_catalog.py` rebuilds the index cleanly (idempotent)
- Re-running `ingest_interactions.py` replaces the existing train/val/test splits
