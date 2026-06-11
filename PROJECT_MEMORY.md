# PROJECT_MEMORY.md
> Persistent memory for the multimodal-fashion-recommender project.
> Updated at the end of every phase. Source of truth for architecture, status, and decisions.

---

## Product Vision

**Sellable B2B SaaS** for Indian fashion retailers (Myntra-scale and D2C: Snitch, Bewakoof, Powerlook).
Core product: multimodal "shop-the-look / similar items / complete-the-outfit" recommendation API
that a retailer embeds in their Shopify store or mobile app.

**Buyers care about:**
- Easy onboarding (catalog CSV or Shopify webhook → live in <30 min)
- Low serving cost / latency (<100 ms P95)
- Measurable business lift (CTR, conversion, AOV)
- White-label: their brand, their data, their domain

**India-first:** INR pricing, Indian sizing system, regional/seasonal signals, Shopify /products.json ingestion.

---

## Architecture

### Core Model
```
User Tower                         Item Tower
──────────────────────────────     ──────────────────────────────
last-20 item sequence              target item
  ↓                                  image (512) + text (384)
item embeddings (20 × 256)            ↓
  ↓                                Concat → (896)
Positional encoding                   ↓
  ↓                                MLP: 896→512→256
2-layer Transformer (4 heads)         ↓
  ↓                                L2-normalize
Masked mean-pool                   256-dim unit vector
  ↓
L2-normalize
256-dim unit vector
               ↘   ↙
           Cosine similarity
           In-batch InfoNCE (τ=0.1)
```
- Image encoder: CLIP ViT-B/32 → 512-dim (frozen)
- Text encoder: SBERT all-MiniLM-L6-v2 → 384-dim (frozen)
- Item fusion: MLP (896→512→256), trained
- User tower: 2-layer Transformer + mean-pool, trained
- Retrieval: FAISS IndexFlatIP, 256-dim
- LLM explanations: Groq llama-3.1-8b-instant (free tier) with template fallback

### Key Hyperparameters (locked — training complete)
| Param | Value |
|-------|-------|
| output_dim | 256 |
| batch_size | 256 (in-batch negatives) |
| LR | 3e-4 |
| warmup_steps | 500 |
| temperature τ | 0.1 |
| seq_len | 20 |
| dropout | 0.2 |
| seed | 42 |

---

## Repository Layout

```
multimodal-fashion-recommender/
├── src/                    # Core Python package
│   ├── data/               # loader.py, preprocess.py
│   ├── encoders/           # image_encoder.py (CLIP), text_encoder.py (SBERT)
│   ├── models/             # item_tower.py, user_tower.py, two_tower.py
│   ├── training/           # dataset.py, train.py, evaluate.py
│   ├── retrieval/          # faiss_index.py
│   └── reasoning/          # llm_explainer.py (Ollama), groq_explainer.py (Groq)
├── app/
│   ├── api/                # FastAPI app (Phase 1)
│   │   ├── main.py         # FastAPI lifespan, mounts /metrics, includes router
│   │   ├── routes.py       # recommend / similar / health endpoints
│   │   ├── auth.py         # X-Api-Key dependency, per-brand
│   │   ├── schemas.py      # pydantic request/response models (includes request_id)
│   │   ├── metrics.py      # Prometheus collectors
│   │   └── logging_config.py  # structlog (PrintLoggerFactory, no add_logger_name)
│   ├── brands/
│   │   └── registry.py     # BrandConfig, BrandState, BrandRegistry, load_registry()
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── schema.py       # Pydantic CatalogRow + InteractionRow models
│   │   ├── sources.py      # CsvSource + ShopifySource (paginated)
│   │   ├── images.py       # Resumable download, retry/backoff, failure manifest
│   │   ├── pipeline.py     # Catalog ingest: fetch→encode→index→brand yaml
│   │   └── interactions.py # Interaction ingest: load, map, split, write parquets
│   └── streamlit_app.py    # Local demo (Ollama)
├── brands/
│   └── h_and_m.yaml        # H&M brand config
├── spaces/                 # HuggingFace Spaces deployment (Groq)
│   ├── app.py
│   └── src/                # Duplicate modules (spaces-specific)
├── scripts/                # Pipeline scripts (00–06) + Phase 2 ingestion CLIs
│   ├── ingest_catalog.py   # Shopify/CSV → embeddings → FAISS index → brand yaml
│   └── ingest_interactions.py  # Orders/events CSV → train/val/test parquets
├── tests/                  # test_models.py, test_encoders.py, test_api_contract.py,
│   │                       #   test_auth.py, test_ingest_*.py, test_image_download.py
│   └── conftest.py         # mock_brand_state, mock_registry, api_client fixtures
├── data/
│   ├── h-and-m-personalized-fashion-recommendations/  ← 30 GB raw
│   └── processed/          ← pre-computed artifacts (FAISS, .npy, .parquet)
├── checkpoints/            # best.pt (multimodal), text_only.pt (text-only baseline)
├── Dockerfile              # Multi-stage, non-root (appuser uid 1001)
├── pyproject.toml          # uv project config with cu128 torch for win32
├── config.yaml             # All training/runtime config
└── brands/h_and_m.yaml     # Brand config (api_key_env points to HM_API_KEY)
```

---

## Phase 1 — API Contract (locked 2026-06-07)

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/{brand}/recommend` | X-Api-Key | User/item → ranked items + explanations |
| GET | `/v1/{brand}/item/{item_id}/similar` | X-Api-Key | Item-to-item (cold-start path) |
| GET | `/health` | none | Liveness + loaded brands |
| GET | `/metrics` | none | Prometheus metrics (ASGI sub-app) |

### Request: POST /v1/{brand}/recommend
```json
{
  "user_id": "string | null",
  "item_id": "string | null",
  "k": 10,
  "explain": false
}
```
- At least one of `user_id` or `item_id` must be provided (pydantic model_validator)
- `k`: 1–100, default 10
- `explain`: if true, calls LLM explainer per result

### Response: POST /v1/{brand}/recommend
```json
{
  "request_id": "uuid4-string",
  "brand": "h_and_m",
  "results": [
    {
      "item_id": "string",
      "score": 0.82,
      "explanation": "string | null",
      "pdp_url": "string | null"
    }
  ],
  "cold_start": false,
  "latency_ms": 41.2
}
```
- `request_id`: server-generated UUID v4. Locked into contract now for Phase 5 click attribution. Never null.
- `cold_start`: true when user_id had no history and fell back to item_id embedding

### Response: GET /v1/{brand}/item/{item_id}/similar
```json
{
  "request_id": "uuid4-string",
  "brand": "h_and_m",
  "query_item_id": "string",
  "results": [
    {
      "item_id": "string",
      "score": 0.91,
      "explanation": null,
      "pdp_url": null
    }
  ],
  "latency_ms": 12.1
}
```
- Self is excluded from results (searches k+1, drops matching item_id)

### Auth
- Header: `X-Api-Key: <key>`
- Missing key → 401
- Wrong key → 401
- Unknown brand in URL → 404
- Key value read from env var named in `api_key_env` field of brand YAML

### Error behavior
| Condition | HTTP |
|-----------|------|
| Missing X-Api-Key header | 401 |
| Wrong key value | 401 |
| Unknown brand slug in URL | 404 |
| Neither user_id nor item_id in body | 422 |
| item_id not in FAISS index | 404 |

---

## Brand Config Schema (`brands/<brand>.yaml`)

```yaml
brand: h_and_m                             # slug used in URL path /v1/{brand}/
display_name: "H&M"                        # human-readable name in /health
catalog_path: data/processed/articles.parquet
index_path: data/processed/faiss_index_active  # FaissRetriever.load() path
transactions_dir: data/processed           # dir with train.parquet/val.parquet/test.parquet
checkpoint_path: checkpoints/best.pt
api_key_env: HM_API_KEY                    # env var name — key value NEVER stored in yaml
embeddings_path: indices/h_and_m/item_emb.npy  # optional; written by ingest_catalog.py
pdp_url_template: null                     # optional; PDP URL from catalog rows directly
llm:
  provider: groq                           # groq | ollama | template
  model: llama-3.1-8b-instant
  enabled: true
# After ingest_interactions.py:
# transactions_dir: data/h_and_m/transactions
```

**Multi-brand-per-process**: All `brands/*.yaml` files loaded at startup. URL path `{brand}` dispatches to the matching BrandState. Different from the reference project (agentic-shopping-assistant) which uses BRAND env var for single-brand-per-process.

---

## Ingestion CSV Schemas

### Catalog CSV (required columns)
```csv
product_id,title,description,image_url,price_inr,category,pdp_url
SN001,"Oversized Tee","100% cotton drop-shoulder",https://cdn.example.com/t.jpg,1299,topwear,https://snitch.co.in/products/oversized-tee
```
| Column | Type | Constraint |
|--------|------|-----------|
| `product_id` | str | non-empty, unique |
| `title` | str | non-empty |
| `description` | str | non-empty |
| `image_url` | str | non-empty URL |
| `price_inr` | float | > 0 |
| `category` | str | non-empty |
| `pdp_url` | str | non-empty URL |

article_id in the output parquet is a sequential integer (1, 2, 3…). The original product_id is preserved as a separate column.

### Interactions CSV — Generic format
```csv
user_id,product_id,timestamp,event_type
u1,SN001,2025-11-01T10:00:00Z,purchase
```
`event_type` must be one of: `purchase`, `view`, `wishlist`, `cart`.

### Interactions CSV — Shopify orders export
Required columns from Shopify admin export: `Email`, `Paid at`, `Lineitem sku`.
All mapped to generic format internally (`Email` → `user_id`, `Paid at` → `timestamp`, `Lineitem sku` → `product_id`, event_type hardcoded to `purchase`).

## Prometheus Metrics

| Metric | Type | Labels | Added |
|--------|------|--------|-------|
| `fashion_rec_requests_total` | Counter | brand, endpoint, status | Phase 1 |
| `fashion_rec_request_latency_seconds` | Histogram | brand, endpoint | Phase 1 |
| `fashion_rec_llm_cost_usd_total` | Counter | brand, provider | Phase 1 |
| `fashion_rec_llm_calls_total` | Counter | brand, provider, status | Phase 1 |
| `fashion_rec_llm_call_duration_seconds` | Histogram | brand, provider | Phase 4 |
| `fashion_rec_llm_tokens_total` | Counter | brand, provider | Phase 4 |
| `fashion_rec_explanation_cache_hits_total` | Counter | brand | Phase 4 |
| `fashion_rec_explanation_cache_misses_total` | Counter | brand | Phase 4 |

Buckets: [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0]

---

## structlog Fields (per request)

```
brand, request_id, latency_ms, usd_cost, inr_cost, cache_hits, cache_misses, cold_start, k, n_results
```

LLM cost estimate: ($0.05×120 + $0.08×80)/1M ≈ $0.0000094 per call (Groq llama-3.1-8b-instant).

---

## Torch Stack

**Windows dev**: `torch==2.11.0+cu128` (from `https://download.pytorch.org/whl/cu128`, driver CUDA 13.x)
**Docker/Linux**: `torch==2.12.0` CPU-only (from PyPI, no platform marker)

Platform dispatch in `pyproject.toml`:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
```

**IMPORTANT**: Do not change the torch source without re-running the eval harness and confirming Recall@10 ≈ 0.0328.

---

## What's Done

| Component | Status | Evidence |
|-----------|--------|---------|
| Two-tower model (item + user towers) | ✅ Complete | 29 passing tests |
| CLIP image encoder (frozen) | ✅ Complete | test_encoders.py passing |
| SBERT text encoder (frozen) | ✅ Complete | test_encoders.py passing |
| Training loop (warmup, InfoNCE, early-stopping) | ✅ Complete | best.pt epoch 13 |
| Eval metrics (Recall@K, NDCG@K, MRR) | ✅ Complete | evaluate.py |
| FAISS index (full 20k + active 10.5k) | ✅ Complete | data/processed/faiss_index*/ |
| LLM explanations (Ollama local + Groq API + template fallback) | ✅ Complete | groq_explainer.py |
| 6-phase pipeline scripts (00–05) + quality gate (06) | ✅ Complete | end-to-end tested |
| Local Streamlit demo | ✅ Complete | app/streamlit_app.py |
| HuggingFace Spaces deployment | ✅ Live | spaces/app.py |
| H&M dataset processed (20k items, 913k train interactions) | ✅ Complete | data/processed/ |
| FastAPI app (recommend / similar / health / metrics) | ✅ Complete | app/api/ |
| Multi-brand registry (BrandRegistry, BrandState) | ✅ Complete | app/brands/registry.py |
| Per-brand X-Api-Key auth | ✅ Complete | app/api/auth.py |
| Pydantic schemas with request_id | ✅ Complete | app/api/schemas.py |
| Prometheus /metrics | ✅ Complete | app/api/metrics.py |
| structlog JSON logging | ✅ Complete | app/api/logging_config.py |
| Cold-start fallback (item embedding when no user history) | ✅ Complete | app/api/routes.py |
| Brand YAML config (h_and_m.yaml) | ✅ Complete | brands/h_and_m.yaml |
| Multi-stage Docker image (non-root, uv sync --frozen) | ✅ Complete | Dockerfile |
| API contract tests + auth tests | ✅ Complete | tests/test_api_contract.py, test_auth.py |
| Catalog ingestion pipeline (Shopify/CSV → CLIP+SBERT → FAISS → brand yaml) | ✅ Complete | app/ingestion/pipeline.py, scripts/ingest_catalog.py |
| ShopifySource (paginated, robots.txt warn-and-proceed) + CsvSource | ✅ Complete | app/ingestion/sources.py |
| Resumable image download (retry/backoff, failure manifest) | ✅ Complete | app/ingestion/images.py |
| Interaction ingestion (generic events CSV + Shopify orders) → train/val/test parquets | ✅ Complete | app/ingestion/interactions.py, scripts/ingest_interactions.py |
| Pydantic catalog + interaction row validation | ✅ Complete | app/ingestion/schema.py |
| CLIENT_ONBOARDING.md | ✅ Complete | CLIENT_ONBOARDING.md |
| Phase 2 tests: 46 ingest + 17 interaction tests | ✅ Complete | 91 passing (vs 73 pre-Phase 2) |
| Re-rank layer over FAISS (pure-Python, per-brand price + category affinity) | ✅ Complete | app/rerank.py |
| Before/after similarity eval (strict + affinity cat-match, guardrails) | ✅ Complete | scripts/eval_similarity_quality.py |
| 133 tests pass (6 /similar tests fixed, rerank mock wiring) | ✅ Complete | tests/ |

---

## Benchmark Results (Phase 0.5 — confirmed 2026-06-07)
> Temporal test split. Script: `scripts/06_baseline_quality_gate.py`. GPU: RTX 3070.
> Note: README previously reported Recall@10=0.041 — that was the **val** metric captured during training
> on the full 20k pool. The numbers below are the rigorous **test** set evaluation.

| Model | Pool | Recall@10 | NDCG@10 | MRR | R@10 lift vs. pop | R@10 lift vs. copurchase |
|-------|------|-----------|---------|-----|-------------------|--------------------------|
| Popularity | full | 0.0107 | 0.0048 | 0.0031 | 1.00× | — |
| Co-purchase (item-item CF) | full | 0.0155 | 0.0081 | 0.0059 | 1.45× | 1.00× |
| Text-only two-tower | full | 0.0211 | 0.0128 | 0.0103 | 1.96× | 1.36× |
| **Multimodal two-tower** | **full** | **0.0299** | **0.0191** | **0.0158** | **2.79×** | **1.93×** |
| Popularity | active | 0.0107 | 0.0048 | 0.0031 | 1.00× | — |
| Co-purchase (item-item CF) | active | 0.0155 | 0.0081 | 0.0059 | 1.45× | 1.00× |
| Text-only two-tower | active | 0.0248 | 0.0151 | 0.0121 | 2.31× | 1.60× |
| **Multimodal two-tower** | **active** | **0.0328** | **0.0208** | **0.0172** | **3.06×** | **2.12×** |

**GATE: ✅ PASS** — Multimodal Recall@10 (active) = 0.0328 vs. co-purchase 0.0155 → **2.12× lift** (threshold ≥1.5×). Proceed to Phase 1.

Additional notes from quality gate run:
- Active pool: 10,556 items. All 4,911 test target items present in both full and active pools (100% coverage).
- Co-purchase coverage: 99.7% of test users had signal; 0.3% fell back to popularity ranking.
- Train: 2020-01-16 → 2020-07-29 | Val: 2020-07-29 → 2020-08-23 | Test: 2020-08-23 → 2020-09-22. No leakage confirmed.

## Eval Re-validation on torch 2.11 serving stack (2026-06-07)

**Context:** After switching from old CUDA 2.4.x torch to `torch 2.11.0+cu128`, re-ran the eval harness to confirm the query encoder change didn't degrade the validated numbers.

**Result:** ✅ **Lift confirmed on torch 2.11 serving stack**
- mm_r_active = 0.03281094302020111 (Δ = 0.000000 vs baseline, identical)
- lift = 2.1156542056074765 (identical to Phase 0.5 baseline)
- Gate: PASS
- Script: `scripts/06_baseline_quality_gate.py`
- The 3.06× popularity / 2.12× co-purchase claim holds on the shipped inference stack.

---

## Eval Setup (confirmed by Phase 0 audit)

- **Temporal split**: 80/10/10 train/val/test by date. Chronological — NO leakage.
- **Leakage check**: `dataset.py` uses `bisect_left(target_ts)` to strictly exclude target timestamp and all future items from user history. Samples with zero prior history are skipped.
- **Active pool**: `index_article_ids_active.npy` — ~10.5k items that appear in at least one transaction. Full pool is 20k (includes cold/inactive items).
- **Val eval during training**: Uses full 20k pool (harder than active-only). Checkpoint selection on val Recall@10 full pool.
- **Eval comparison script**: Tests both full and active pools for all models.

---

## Known Issues

### Pre-existing test failure: test_dataset_getitem_keys_and_shapes (out of scope)
- **File**: `tests/test_models.py`
- **Symptom**: Dataset `__getitem__` returns a `target_idx` key; test expects exactly 5 keys and fails.
- **Root cause**: Pre-dates Phase 1. Dataset was modified to include `target_idx` but the test was not updated.
- **Status**: Leave as-is. Not caused by any Phase 1 change. 31/32 tests pass.
- **Action**: Fix in Phase 2 or whenever tests/test_models.py is touched for another reason.

---

## What's Missing (Gap vs. Sellable Product)

### P0 — Blockers (resolved in Phase 1)
| Gap | Status |
|-----|--------|
| No REST API (FastAPI) — only Streamlit demo | ✅ Phase 1 complete |
| No multi-tenancy — single H&M catalog hardcoded | ✅ Phase 1 complete |
| No Docker / Cloud Run deploy | ✅ Dockerfile done; Cloud Run in Phase 2+ |
| No cold-start API path | ✅ Phase 1 complete |
| No observability (Prometheus, structlog) | ✅ Phase 1 complete |
| No auth (API keys per brand) | ✅ Phase 1 complete |

### P0 — Still needed
| Gap | Impact |
|-----|--------|
| No Indian brand catalog in this project | Demo not client-presentable to Indian retailers |

### P1 — Required for commercial viability
| Gap | Impact |
|-----|--------|
| ~~No explanation caching~~ | ✅ Phase 4: Redis + LRU fallback |
| ~~No Cloud Run deploy + WIF auth~~ | ✅ Phase 4: deploy.yml workflow_dispatch ready |
| No onboarding automation (CLI flags only; no web UI) | Requires data engineer; not self-serve |

### P2 — For scale / enterprise sales
| Gap | Impact |
|-----|--------|
| No A/B testing / champion-challenger | Can't prove lift to clients |
| No online learning / feedback loop | Model doesn't improve from clicks |
| No click/impression Postgres logging | No click attribution (request_id ready, storage not wired) |

---

## Sibling Project: agentic-shopping-assistant

**Location:** `C:\Users\gaura\ml-projects\agentic-shopping-assistant`

### Brand Config Pattern (ported in Phase 1)

**Files ported (adapted):**
- `src/config/brand.py` → `app/brands/registry.py` (BrandConfig, BrandState, BrandRegistry)
- `api/main.py` → `app/api/main.py` (lifespan, load_registry)
- `Dockerfile` → root Dockerfile (multi-stage, uv, non-root)

**Key difference from reference:** Reference uses `BRAND` env var → single brand per process.
This project loads ALL `brands/*.yaml` at startup → multi-brand-per-process with URL path routing.

**CI/CD:** `.github/workflows/deploy-demo.yml` — matrix deploys same Docker image to Cloud Run N times with different `BRAND` + `INDEX_STORE_URI` env vars. Region: `asia-south1`. WIF auth (no service account keys).

---

## Indian Brand Catalog Data (available in sibling project)

Source: `C:\Users\gaura\ml-projects\agentic-shopping-assistant\data\raw\`

| Brand | BRAND= | Source | Items | CSV Path | PDP URL Template | Verified |
|-------|--------|--------|-------|----------|------------------|---------|
| Snitch | snitch | Shopify live | 15,001 | `shopify/snitch/products.csv` | `https://snitch.co.in/products/{handle}` | ✅ HTTP 200 |
| Fashor | fashor | Shopify live | 3,619 | `shopify/fashor/products.csv` | `https://fashor.com/products/{handle}` | ✅ HTTP 200 |
| Virgio | virgio | Shopify live | 1,811 | `shopify/virgio/products.csv` | `https://virgio.com/products/{handle}` | ✅ HTTP 200 |
| Powerlook | powerlook | Shopify live | 928 | `shopify/powerlook/products.csv` | `https://powerlook.in/products/{handle}` | ✅ HTTP 200 |
| Myntra | myntra | Kaggle | 14,463 | `myntra/Fashion Dataset.csv` | `https://www.myntra.com/{handle}` | ❌ No live PDP |
| Flipkart | flipkart | Kaggle | 16,000 | `flipkart/fashion_products.csv` | `{handle}` (raw URL in data) | ✅ HTTP 200 |

**Catalog fields (internal schema, all brands):** `article_id`, `prod_name`, `product_type_name`, `product_group_name`, `colour_group_name`, `department_name`, `detail_desc`, `price_inr`, `size_system`, `pdp_handle`, `image_url`

**For Phase 3 demo:** Use Snitch (15k items, verified PDP URLs, Indian men's streetwear — high visual differentiation, good for multimodal demo). Fashor as second brand (Indian women's ethnic wear, complements Snitch).

**Ingestion path (Phase 2 → 3):** Port `src/catalogue/adapter.py` from sibling project to handle Shopify CSV → internal schema mapping, then run through Phase 1 embedding + index pipeline.

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Pre-phase-0 | Frozen CLIP + SBERT encoders | Cost; fine-tuning adds <2% lift at 10× training cost |
| Pre-phase-0 | IndexFlatIP over HNSW/IVF | Correctness first; 20k items fits in RAM |
| Pre-phase-0 | Groq llama-3.1-8b-instant for explanations | Free tier, fast, good enough for fashion descriptions |
| Pre-phase-0 | H&M dataset for training | Largest public fashion interaction dataset (31M tx) |
| Pre-phase-0 | Warmup + τ=0.1 + LR=3e-4 | Fixes representation collapse (MLP saturation) |
| 2026-06-07 | Insert Phase 0.5 (quality gate) before Phase 1 | Model must beat co-purchase baseline (real retail bar) before investing in serving infra |
| 2026-06-07 | Port brand-config pattern from agentic-shopping-assistant | Consistency across portfolio; pattern already proven; avoids reinventing YAML schema |
| 2026-06-07 | Use sibling project's Indian brand catalog data for Phase 3 | Verified PDP URLs already exist; avoids re-scraping; Snitch + Fashor cover men's + women's Indian fashion |
| 2026-06-07 | Multi-brand-per-process (not BRAND env var) | Single Cloud Run instance serves all brands; cleaner for portfolio demo; URL path routing |
| 2026-06-07 | Add request_id (UUID4) to all API responses in Phase 1 | Locks API contract now so Phase 5 click attribution doesn't force API re-version |
| 2026-06-07 | Switch torch CUDA index cu124→cu128 (torch 2.11+) | transformers≥4.52 uses torch.float8_e8m0fnu, absent in torch 2.6; cu128 has it |
| 2026-06-07 | Use FAISS index.reconstruct(row) for user sequence embeddings | Avoids loading 400MB+ raw CLIP/SBERT arrays at API runtime; 256-dim item embeddings already stored in FAISS |
| 2026-06-07 | structlog PrintLoggerFactory (no add_logger_name) | add_logger_name requires .name attribute; PrintLogger has none — AttributeError on every log call |
| 2026-06-08 | Sequential integer article_ids in FAISS (not original product_ids) | Existing registry code uses `int(aid)`; sequential IDs preserve compatibility. Original product_id preserved in catalog parquet as separate column. |
| 2026-06-08 | robots.txt warn-and-proceed by default; --respect-robots opt-in | Client is ingesting their OWN store under authorized agreement; hard-stop would block legitimate onboarding. Flag exists for strict-compliance use. |
| 2026-06-08 | Lazy-import heavy ML deps in pipeline.py via private helper functions | open_clip is in [ml] optional extra and absent from the test conda env. Lazy helpers (_build_img_encoder etc.) are patchable by name, avoid import-time failures, and don't change the patching contract for tests. |
| 2026-06-08 | requests (not httpx) for image download | requests was already in runtime deps; httpx would add a new dep. ThreadPoolExecutor provides the concurrency; async wasn't needed for a write-once pipeline. |

---

## Open Questions

1. **FAISS index type at scale** — At >100k items, switch from IndexFlatIP to IVF_FLAT or HNSW for <10 ms latency?
2. **Serving architecture** — Single-tenant Cloud Run per brand, or multi-tenant with namespace isolation in one deployment?
3. **Groq rate limits** — Free tier is 30 req/min; need Redis cache or upgrade to paid for production traffic
4. **Online learning** — Should feedback loop retrain full model, or just fine-tune user tower?
5. **Catalog adapter** — Port the sibling project's `catalogue/adapter.py` as-is or rewrite for this project's internal schema?

---

## Phase History

| Phase | Title | Status | Date |
|-------|-------|--------|------|
| Phase 0 | Audit + Architecture Map | ✅ Complete | 2026-06-07 |
| **Phase 0.5** | **Model Quality Gate (baseline table + lift verdict)** | ✅ Complete — PASS (2.12× lift) | 2026-06-07 |
| **Phase 1** | **Production API + Multi-Tenancy** | ✅ Complete | 2026-06-07 |
| **Phase 2** | **Catalog Ingestion + Interaction Ingestion** | ✅ Complete | 2026-06-08 |
| Phase 3 | Indian Brand Demo (Snitch + Fashor via Phase 2 pipeline) | — | — |
| **Phase 4** | **Deploy + Cost + Caching** | ✅ Complete | 2026-06-09 |
| **Phase 5 (re-rank)** | **Similarity Re-rank + Quality Eval** | ✅ Complete | 2026-06-11 |
| Phase 5 (A/B) | Champion-Challenger + Online Learning | — | — |

### Phase 0.5 — Exit Criteria (ALL MET ✅)
- [x] Popularity baseline numbers confirmed on temporal test split, active pool AND full pool
- [x] Co-purchase (item-item co-occurrence) baseline implemented and evaluated on same splits
- [x] Text-only two-tower evaluated on same splits
- [x] Multimodal two-tower re-confirmed on same splits
- [x] Lift multipliers computed (model / each baseline) for Recall@10, NDCG@10
- [x] Eval leakage sanity-check documented (train/val/test date ranges confirmed, no overlap)
- [x] GATE: multimodal Recall@10 (active) = 0.0328 vs. copurchase 0.0155 → 2.12× ≥ 1.5× → PASS
- Script: `scripts/06_baseline_quality_gate.py` | Results: `data/processed/quality_gate_results.json`

### Phase 1 — Exit Criteria (ALL MET ✅)
- [x] README headline metric corrected (0.041 val → test numbers 0.0299 full / 0.0328 active)
- [x] README baseline+lift table present (Phase 0.5 comparison table)
- [x] `POST /v1/{brand}/recommend` returns ranked items with request_id, cold_start flag
- [x] `GET /v1/{brand}/item/{item_id}/similar` returns item-to-item results (cold-start path)
- [x] `GET /health` lists loaded brands
- [x] `GET /metrics` exposes labelled Prometheus counters
- [x] Wrong/missing API key → 401; wrong brand → 404
- [x] Brand config: `brands/h_and_m.yaml` + BrandConfig pydantic schema
- [x] Per-brand FAISS index + embeddings loaded at startup
- [x] Cold-start fallback: item embedding when user has no history
- [x] Prometheus /metrics with brand, endpoint, status labels
- [x] structlog JSON logging with brand, request_id, latency_ms, usd_cost
- [x] Multi-stage Docker image (non-root appuser uid 1001, uv sync --frozen)
- [x] `docker build -t fashion-rec:test .` → ✅ PASS (2.9 GB image)
- [x] 31/32 tests pass (1 pre-existing failure: test_dataset_getitem_keys_and_shapes, out of scope)
- [x] Eval re-validation: Recall@10 = 0.0328 on torch 2.11 serving stack (Δ=0.000000 vs baseline)
- [x] PROJECT_MEMORY.md updated with API contract + brand-config schema

### Phase 2 — Exit Criteria (ALL MET ✅)
- [x] `scripts/ingest_catalog.py` accepts Shopify /products.json URL or CSV path → builds brand index
- [x] `scripts/ingest_interactions.py` accepts Shopify orders + generic events CSV → user sequences
- [x] Cold-start path: item-to-item similarity works with zero interaction data
- [x] `CLIENT_ONBOARDING.md` written for client data engineers
- [x] Integration test: fixture Snitch CSV → FAISS index queryable via Phase 1 API (TestPipelineIntegration)
- [x] 91 tests passing (46 new ingest tests + 17 interaction tests + 28 pre-existing Phase 1 tests)
- [x] Phase 2 code passes `ruff check` (scoped to Phase 2 files)

### Phase 3 — Exit Criteria (revised)
- Snitch + Fashor ingested via Phase 2 pipeline (source: sibling project CSVs)
- Both brands queryable via Phase 1 API with verified PDP URLs
- Demo script updated for Indian brand results (kurta, ethnic wear categories visible)
- Spaces deployment updated with Indian brand dropdown

### Phase 4 — Exit Criteria (ALL MET ✅)
- [x] `app/pricing.py` — Groq cost constants, Cloud Run rates, `groq_call_cost_usd()`, `usd_to_inr()`, `cost_per_1000_recommendations()`
- [x] `app/cache.py` — ExplanationCache (Redis primary, in-process LRU fallback); graceful Redis failure → LRU; no crash when Redis absent
- [x] `app/storage.py` — GCS brand asset sync at startup; local-path no-op when `GCS_BUCKET_NAME` unset; mocked in CI (zero real GCS calls)
- [x] Per-call structured logs: `usd_cost`, `inr_cost`, `cache_hits`, `cache_misses`
- [x] New Prometheus metrics: `fashion_rec_llm_call_duration_seconds`, `fashion_rec_llm_tokens_total`, `fashion_rec_explanation_cache_hits_total`, `fashion_rec_explanation_cache_misses_total`
- [x] `infra/Dockerfile.cpu` — CPU-only image (no CUDA runtime); respects `PORT` env var; healthcheck on `/health`; non-root (uid 1001)
- [x] `infra/docker-compose.yml` — API + Redis for local dev; brand assets mounted from host
- [x] `.github/workflows/deploy.yml` — `workflow_dispatch` only; WIF auth (no SA key files); build → push → Cloud Run; all secrets via GitHub Secrets
- [x] `COST.md` — cost-per-1000 derivation with 4 scenarios; assumptions documented
- [x] GCP project: user-provisioned, NOT aetherart-497918; referenced via `${{ secrets.GCS_PROJECT }}`; bucket via `GCS_BUCKET_NAME` env var
- [x] 133 tests pass (15 new tests vs Phase 3 baseline); 1 pre-existing known failure unchanged
- [x] DRAFT PR opened from branch `phase-4-deploy-cost-caching`

### Phase 5 (similarity re-rank) — Exit Criteria (ALL MET ✅ — 2026-06-11)
- [x] `app/rerank.py` — pure-Python re-ranker; score = 0.70×sim - w_price×price_penalty + 0.15×category_affinity
- [x] Per-brand `rerank:` stanza in brands/*.yaml with price weights + category-equivalence groups
- [x] `/similar` wired to rerank layer behind `rerank.enabled` toggle; `BrandConfig` carries `RerankConfig`
- [x] `scripts/eval_similarity_quality.py` — before/after eval; reports strict (exact) + affinity (≥0.4) cat-match + mean |ΔPrice| per pass
- [x] All per-brand guardrails pass at n=100: Fashor strict≥58%, Fashor |ΔPrice| drops, Powerlook strict≥64%
- [x] Test suite: 133 pass (was 127; 6 `/similar` tests fixed), 1 pre-existing failure unchanged
- [x] DRAFT PR open from `phase-5-similarity-eval`

### Phase 5 (A/B + online learning) — Exit Criteria (ORIGINAL — deferred)
- Champion-challenger traffic split in brand config
- Click/impression logging to Postgres (request_id → impression; click event links back)
- `evals/` harness with Recall@K fixtures per brand
- Weekly eval report auto-generated

---

## Phase 3 — Indian Brand Demo (complete)

**Status:** Complete, merged to main via draft PR.

### Brands ingested

| Brand | Items | Catalog path | Index path | PDP base |
|---|---|---|---|---|
| Snitch | 500 (stratified) | data/snitch/items.parquet | indices/snitch/active.faiss | https://snitch.co.in/products/ |
| Fashor | 3,618 (full) | data/fashor/items.parquet | indices/fashor/active.faiss | https://fashor.com/products/ |
| Powerlook | 906 (full) | data/powerlook/items.parquet | indices/powerlook/active.faiss | https://powerlook.in/products/ |

### Data provenance
- Source: `C:\Users\gaura\ml-projects\agentic-shopping-assistant\data\processed\{brand}\catalogue.parquet`
- Transform script: `scripts/prepare_indian_catalogs.py` (parquet → 7-column CSV)
- Ingest: `scripts/ingest_catalog.py --source csv` (unchanged Phase 2 pipeline)
- PDP URLs: all 15 sampled URLs verified HTTP 200 before ingest

### Synthetic users
- 15 users per brand (45 total), 8–15 interactions each
- user_id format: `synthetic_{brand}_{archetype}_{n}` — labelled synthetic everywhere
- Generator: `scripts/generate_synthetic_users.py`
- Ingested via `scripts/ingest_interactions.py` (transactions at data/{brand}/transactions/)
- NOT personalization quality — illustrative only, as documented in README and demo UI

### Transfer-learning honesty
- Item tower (CLIP + SBERT → MLP) transfers cleanly to any catalog → /similar is the hero flow
- User tower trained on H&M sequences → does NOT transfer to fresh Indian catalogs
- /recommend on synthetic users = illustrative only, labelled in UI with warning banner

### Demo flows
1. **More Like This** (`/v1/{brand}/item/{id}/similar`) — strong day-one content story
2. **Personalized Recs [Illustrative]** (`/v1/{brand}/recommend`) — labelled synthetic, API surface demo

### Files added / changed
- `scripts/prepare_indian_catalogs.py` — parquet-to-CSV catalog prep
- `scripts/generate_synthetic_users.py` — synthetic interaction generator
- `app/streamlit_app.py` — brand selector + Indian brand flows (H&M path unchanged)
- `app/ingestion/interactions.py` — bug fix: `dtype={"product_id": str}` in load_generic_csv
- `tests/test_indian_demo.py` — 22 smoke tests (113 total, 0 regressions)
- `brands/snitch.yaml`, `brands/fashor.yaml`, `brands/powerlook.yaml` — brand configs
- `README.md` — multi-brand intro, Indian brand section, transfer-learning note
- `demo/demo_script.md` — salesperson walkthrough

### Brand API key env vars (demo)
- `SNITCH_API_KEY` → set to any value for local demo
- `FASHOR_API_KEY` → set to any value for local demo
- `POWERLOOK_API_KEY` → set to any value for local demo

---

## Phase 4 — Deploy + Cost + Caching (complete 2026-06-09)

**Status:** Complete. DRAFT PR open from `phase-4-deploy-cost-caching`.

### New modules
| Module | Purpose |
|--------|---------|
| `app/pricing.py` | Groq + Cloud Run cost constants; `groq_call_cost_usd()`, `usd_to_inr()`, `cost_per_1000_recommendations()` |
| `app/cache.py` | ExplanationCache (Redis → LRU fallback); `make_key()` keyed by brand+user_hist_ids+item_id+cold_start; TTL=3600s |
| `app/storage.py` | GCS brand asset sync; mirrors repo-relative paths into bucket; no-op when `GCS_BUCKET_NAME` unset |

### Infrastructure added
| File | Purpose |
|------|---------|
| `infra/Dockerfile.cpu` | CPU-only image; no uv.lock (re-resolves on Linux = CPU torch from PyPI); non-root, `PORT`-aware, healthcheck |
| `infra/docker-compose.yml` | API + Redis; brand data mounted from host |
| `.github/workflows/deploy.yml` | `workflow_dispatch` → build/push/deploy to Cloud Run; WIF auth; all secrets via GitHub Secrets |
| `COST.md` | Cost-per-1000 derivation; 4 scenarios; assumptions documented |

### Cost summary (worst case)
| Scenario | Cost / 1k recs (USD) | Cost / 1k recs (INR) |
|----------|----------------------|----------------------|
| explain=true, no cache | $0.0124 | ₹1.03 |
| explain=true, 80% cache hit | $0.00248 | ₹0.21 |
| explain=false (retrieval only) | $0.00 | ₹0.00 |

### Deploy checklist (human steps before first Cloud Run run)
1. Create new GCP project (NOT aetherart-497918)
2. Enable: Artifact Registry, Cloud Run, Secret Manager, GCS APIs
3. Create Artifact Registry repo (`fashion-rec`)
4. Create GCS bucket for brand assets; upload indices/checkpoints/parquets mirroring repo layout
5. Create Workload Identity Federation pool + provider for GitHub Actions
6. Create service account with: Artifact Registry Writer, Cloud Run Developer, Secret Manager Accessor, Storage Object Viewer
7. Add GitHub Secrets: `GCS_PROJECT`, `GCS_BUCKET_NAME`, `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `SNITCH_API_KEY`, `FASHOR_API_KEY`, `POWERLOOK_API_KEY`, `GROQ_API_KEY`
8. Add GitHub Variables: `GCP_REGION`, `GAR_HOSTNAME`, `GAR_REPO`
9. Run workflow: Actions → Deploy to Cloud Run → Run workflow → staging
10. Verify `/health` returns OK; verify `/metrics` exposes all 8 metrics

### Docker build (pending human action)
- `docker build -f infra/Dockerfile.cpu -t fashion-rec:cpu .` — requires Docker Desktop running
- Before/after size comparison pending (baseline: 2.9 GB; expected: ~1–1.5 GB CPU-only)

### Decisions made in Phase 4
| Decision | Rationale |
|----------|-----------|
| LRU fallback for cache (no Redis in local dev) | Explanation caching works with zero infra in dev; Redis is opt-in via `REDIS_URL` |
| `workflow_dispatch` only deploy | Unreviewed code must not auto-deploy to a public endpoint |
| No `uv.lock` in Dockerfile.cpu | Windows lock has CUDA wheels; re-resolving on Linux gives CPU torch from PyPI |
| GCS object key == local relative path | Simplest mapping; no translation layer; bucket mirrors repo layout |
| In-process LRU cache (not functools.lru_cache) | OrderedDict LRU is evictable and testable; functools.lru_cache is not evictable and hides the key logic |

---

## Phase 5 — Similarity Re-rank (complete 2026-06-11)

**Branch:** `phase-5-similarity-eval` | **PR:** DRAFT open

### New modules
| Module | Purpose |
|--------|---------|
| `app/rerank.py` | Pure-Python re-ranker. `score = 0.70×sim - w_price×price_penalty + 0.15×cat_affinity`. `CategoryAffinityMap` builds three-tier affinity (exact 1.0 / equiv-group 0.7 / related-group 0.4 / none 0.0) from per-brand `category_groups`. No model, no GPU. |

### Brand re-rank config (added to `brands/*.yaml`)
| Brand | w_price | price_norm_inr | Category groups |
|-------|---------|----------------|-----------------|
| Fashor | 0.25 | ₹1,200 | ethnic_set (3P/2P/Kurta Set/Co-ord), ethnic_individual (Kurtas/Kurta/Kurti), western_dress (Dresses); ethnic_set↔ethnic_individual = related (0.4) |
| Snitch | 0.15 | ₹500 | bottoms (Trousers/Jeans/Cargo Pants) |
| Powerlook | 0.10 | ₹400 | none (category_groups: []) |

### Before/After Eval Results (n=100 queries/brand, k=5, seed=42 — LOCKED)

> **Strict is the headline metric.** Affinity is reported alongside it so taxonomy-driven improvement is auditable separately from genuine retrieval improvement. Affinity threshold = 0.4 (counts exact + equivalent-group + related-group).

| Brand | Strict raw | Strict rkd | Affinity raw | Affinity rkd | \|ΔPrice\| raw | \|ΔPrice\| rkd | Guardrail |
|-------|-----------|-----------|-------------|-------------|--------------|--------------|-----------|
| snitch | 72% | **96%** (+24pp) | 77% | 100% (+23pp) | ₹369 | ₹158 (↓) | ✅ OK |
| fashor | 64% | **86%** (+22pp) | 93% | 96% (+3pp) | ₹735 | ₹109 (↓) | ✅ OK |
| powerlook | 76% | **94%** (+18pp) | 76% | 94% (+18pp) | ₹305 | ₹98 (↓) | ✅ OK |

Powerlook affinity == strict because no category groups are defined — all improvement is price-sorting. Fashor raw affinity was already 93% (CLIP visual embeddings cluster ethnic styles loosely) — the strict gap (64%) was the honest pre-rerank baseline.

n=25 → n=100 stability check: reranked strict moved ≤5pp for all brands. Numbers locked at n=100.

### Guardrail results
- Fashor strict 86% ≥ 58% floor ✅
- Fashor |ΔPrice| dropped (₹735 → ₹109) ✅
- Powerlook strict 94% ≥ 64% floor ✅
- Fashor price-improved-but-strict-regressed advisory: **not fired** (both improved together; 0.25 weight is correct)

### Decisions made in Phase 5
| Decision | Rationale |
|----------|-----------|
| Report strict + affinity separately | Affinity looks better partly because we defined the equivalence groups; strict is the only number comparable to any external baseline. Never collapse to just affinity. |
| Affinity threshold = 0.4 (related_group_bonus) | Counts all three affinity tiers. Stated explicitly in eval output so future readers can audit what "affinity match" means. |
| Per-brand price weights (not global) | Fashor items span ₹600–₹4000; Snitch ₹400–₹1500; different normalisation constants make price penalty comparable across catalogs. |
| Pure-Python reranker, no model | Re-rank runs at zero cost, zero latency on CPU. Feature vectors (price, category) are already in the FAISS art_map. No inference path needed. |
