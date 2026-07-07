# Client Onboarding Guide

## Try it before you commit to anything

You don't need to wait for your own catalog to be onboarded to start evaluating this API
or building your integration. There's a public sandbox — a real dataset, the real trained
model, no catalog handoff required:

**See `QUICKSTART.md`** — copy-paste curl/JS examples against the sandbox, ready right now.

The rest of this document covers getting **your own** catalog live.

---

## Honest state of this process today

Getting your catalog into our ingestion pipeline is fully documented below and works as
described — you (or we, on your behalf) can run it right now. **Going live on the shared
production service is currently a short, coordinated process with us, not a fully
self-serve one.** Concretely: after your catalog is ingested, someone on our side syncs the
resulting index to our cloud storage, provisions your API key, and deploys — that part
isn't automated yet. We're actively shortening this (tracked internally); this doc will be
updated with a firm SLA once that lands. In the meantime, treat Step 2 below as "hand this
to your integration contact," not "you're live."

## Prerequisites

- Python 3.11+ (only needed if you're running catalog preparation yourself, rather than
  sending us a catalog export directly)
- One of the following catalog sources:
  - A Shopify store with `/products.json` enabled (most stores have this by default)
  - A CSV export of your product catalog (column requirements below)
- Your integration contact will provision your `<BRAND>_API_KEY` once your catalog is live

---

## Step 1 — Prepare your catalog

This step produces the files your integration contact needs — it does not itself make
anything live. You can run it yourself (fastest — send us the output), or send us your raw
catalog export/Shopify URL and we'll run it.

### If running it yourself

```bash
git clone https://github.com/<org>/multimodal-fashion-recommender.git
cd multimodal-fashion-recommender
pip install uv
uv sync --extra ml
```

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
```

**Files written (local only at this point — see Step 2):**
```
data/your_brand/images/          ← downloaded product images
data/your_brand/items.parquet    ← catalog with article_ids
indices/your_brand/active.faiss/ ← FAISS index (queryable immediately, locally)
indices/your_brand/item_emb.npy  ← 256-dim item embeddings
brands/your_brand.yaml           ← brand config
```

**Resumability:** If the script is interrupted during image download, re-run the same command. Already-downloaded images are skipped. Failed images are recorded in `data/your_brand/images/failed_images.json`.

---

## Step 2 — (Optional) Ingest interaction history

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

**Note:** personalization quality depends on interaction volume. With little to no history,
`/recommend` transparently falls back to item-based cold-start — see `README.md`'s
"Multi-Brand Indian Demo" section for how this was handled for our existing brands.

---

## Step 3 — Hand off to go live

This is the step that isn't self-serve yet. Send your integration contact:
- The `data/your_brand/`, `indices/your_brand/`, and `brands/your_brand.yaml` output from
  Step 1 (or just the raw catalog/Shopify URL if you'd rather we run Step 1)
- Confirmation of whether interaction data (Step 2) is included

We sync the assets to our cloud storage, provision your `<BRAND>_API_KEY`, and deploy. Once
that's done, your integration contact will confirm you're live and share your key.

## Step 4 — Test locally before you have a live key (optional)

If you want to verify your catalog ingested correctly before we deploy it, you can run the
API against your local output:

```bash
export YOUR_BRAND_API_KEY=<any-value-you-choose-for-local-testing>
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -H "X-Api-Key: <your-local-value>" \
  http://localhost:8000/v1/your_brand/item/1/similar?k=5
```

**This is local-only sanity-checking** — it doesn't touch the production service and the
key you set here has no relationship to the real key you'll receive when you're live. Most
integrators can skip this and just review the ingestion script's console output instead.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `RuntimeError: /products.json returned 404` | Store has JSON API disabled | Use `--source csv` instead |
| `ValueError: Generic CSV missing required columns: ['price_inr']` | Missing column in your CSV | Add the column (see column table in Step 1B) |
| `FileNotFoundError: Catalog parquet not found` | Ran `ingest_interactions` before `ingest_catalog` | Run Step 1 first |
| `50%+ of catalog rows invalid` | Widespread data quality issue | Check CSV encoding; ensure `price_inr` > 0 for all rows |
| Images downloading slowly | Large catalog with many images | Add `--workers 16` to `ingest_catalog.py` (default: 8) |
| `ModuleNotFoundError: No module named 'open_clip'` | Installed without `[ml]` extra | Re-install: `uv sync --extra ml` |

## Notes on robots.txt

By default, the Shopify ingestion logs a warning if robots.txt restricts `/products.json` but proceeds anyway — you are ingesting **your own store** under an authorized onboarding agreement. To enforce strict robots.txt compliance (e.g. for third-party stores), pass `--respect-robots`.

---

## What happens once you're live

- `/v1/your_brand/item/{id}/similar` is immediately available (cold-start, no interaction data needed)
- `/v1/your_brand/recommend` with `user_id` requires interaction data from Step 2
- Re-running `ingest_catalog.py` rebuilds the index cleanly (idempotent) — send us the updated output the same way as Step 3
- See `QUICKSTART.md` for the full API reference and integration patterns (including the CORS/proxy requirement — read this before you start building your frontend)
