# Project Spec: multimodal-fashion-recommender — Phase 2 (Catalog Ingestion Pipeline)

## Goal
A new brand goes from "here's our catalog" to "queryable via the Phase 1 API" in one
command, in under 30 minutes, with zero interaction history required. This is the
onboarding story we sell: a retailer points us at their Shopify endpoint or hands us a
CSV, and they're live. Until this exists, every client onboarding is us manually running
six scripts — which is not a product. After this, it is.

## Current state (existing project — DO NOT break)
Phase 0/0.5/1 complete and verified:
- Two-tower model (CLIP image + SBERT text → 256-dim), validated lift 3.06× pop / 2.12×
  co-purchase, re-confirmed on the torch 2.7 serving stack (Δ=0.000000)
- Production FastAPI multi-tenant API: `/v1/{brand}/recommend`, `/v1/{brand}/item/{id}/similar`
  (cold-start path), `/health`, `/metrics`; per-brand `X-Api-Key` auth; `request_id` in
  responses; structlog + Prometheus; Docker image builds
- Brand registry loads `brands/<brand>.yaml` at startup; per-brand namespaced FAISS indices
- 31/32 tests pass (1 known pre-existing dataset test failure)

Load-bearing — orchestrator must NOT change without escalating:
- Model architecture / checkpoint format / encoder code
- Eval harness + validated baseline numbers
- The Phase 1 API contract (routes, schemas incl. `request_id`, auth, brand-config schema)
- FAISS index build logic (ingestion CALLS it; does not reimplement it)
- Existing tests (must keep passing)

Reference (read-only): `C:\Users\gaura\ml-projects\agentic-shopping-assistant` has a
Shopify `/products.json` onboarding path documented in its `CLIENT_ONBOARDING.md` and
ingestion code — read it and reuse the Shopify-pagination + image-handling patterns.

## Scope

### In scope (this iteration)
- `scripts/ingest_catalog.py` — one command, accepts EITHER:
  - `--source shopify --url https://<brand>.com/products.json` (paginated, 250/page)
  - `--source csv --path <file.csv>` (documented column schema)
  Pipeline: fetch products → download images (resumable, rate-limited, retries) → build
  item text → run CLIP+SBERT (reuse existing encoders) → write embeddings → build FAISS
  index (call existing build logic) → write `brands/<brand>.yaml` → all in correct
  namespaced paths so the Phase 1 API serves it immediately with no further steps.
- `scripts/ingest_interactions.py` — OPTIONAL interaction data, accepts:
  - Shopify orders CSV or generic events CSV → interaction format → user sequences
  - If omitted entirely, brand is still fully usable in cold-start / content-only mode
    (item-to-item via `/similar`). Interactions only unlock personalized `/recommend`.
- CSV schema definition (documented + validated): required columns for catalog
  (`product_id`, `title`, `description`, `image_url`, `price_inr`, `category`, `pdp_url`)
  and for interactions (`user_id`, `product_id`, `timestamp`, `event_type`).
- Robustness: resumable image download (skip already-fetched), retry w/ backoff, a
  manifest of failed/missing images, graceful handling of brands that disable
  `/products.json` (clear error → tell them to use CSV path).
- `CLIENT_ONBOARDING.md` — step-by-step: get catalog → run ingest → (optional) ingest
  interactions → API key → test call. Written for a client's data engineer, not us.
- Idempotency: re-running ingest for a brand updates cleanly, doesn't corrupt the index.

### Out of scope (do not build)
- Real Indian brand demo data (that's Phase 3 — this phase ships the *pipeline* + a
  small synthetic fixture to test it)
- Redis cache, Grafana, Sentry, OTel, Cloud Run deploy (Phase 4)
- A/B framework, feedback loop, click logging (Phase 5)
- Any model retraining or architecture change
- Incremental/streaming catalog updates (full rebuild per ingest is fine this phase)

## Tech stack
- Python 3.11+, existing torch 2.7 / faiss / sentence-transformers / transformers stack
- httpx (async image download w/ retries) or requests — orchestrator picks, justify
- pandas/pyarrow for CSV + parquet (already present)
- pydantic v2 for catalog-row + interaction-row validation
- tenacity (or hand-rolled) for retry/backoff — escalate if adding tenacity

## Architecture
```
scripts/
  ingest_catalog.py        # shopify | csv → embeddings → index → brand yaml
  ingest_interactions.py    # optional orders/events csv → user sequences
app/
  ingestion/
    sources.py             # ShopifySource (paginated) + CsvSource, common interface
    images.py              # resumable download, retry/backoff, failure manifest
    schema.py              # pydantic catalog-row + interaction-row models
    pipeline.py            # orchestrates fetch→embed→index→config (calls existing encoders/FAISS)
brands/
  fixture_snitch.yaml      # produced by test fixture ingest
tests/fixtures/
  snitch_catalog.csv       # ~20 rows, Snitch-format, for the ingest test
CLIENT_ONBOARDING.md
tests/
  test_ingest_catalog.py   # csv + (mocked) shopify path → queryable index
  test_ingest_schema.py    # row validation, bad rows rejected with clear errors
  test_image_download.py    # resumability + retry + failure manifest (mocked http)
```

## Data model
```csv
# catalog CSV — required columns
product_id,title,description,image_url,price_inr,category,pdp_url
SN001,"Oversized Tee","100% cotton drop-shoulder",https://...,1299,topwear,https://snitch.co.in/...
```
```csv
# interactions CSV (optional) — required columns
user_id,product_id,timestamp,event_type
u1,SN001,2025-11-01T10:00:00Z,purchase
```
```yaml
# brands/<brand>.yaml written by ingest (api_key_env left for human to set)
brand: snitch
display_name: "Snitch"
catalog_path: data/snitch/items.parquet
index_path: indices/snitch/active.faiss
embeddings_path: indices/snitch/item_emb.npy
pdp_url_template: null        # PDP comes from catalog rows directly
api_key_env: SNITCH_API_KEY
llm: { provider: template, enabled: true }   # safe default; human upgrades to groq
```

## Verification commands
```yaml
- name: tests
  cmd: pytest -v
  required: true
- name: lint
  cmd: ruff check .
  required: true
- name: types
  cmd: mypy app/ scripts/
  required: false
- name: e2e_ingest
  cmd: python scripts/ingest_catalog.py --source csv --path tests/fixtures/snitch_catalog.csv --brand snitch
  required: true
```

## Subagent usage rules
- `executor` writes/edits; `verifier` runs tests/lint/types/e2e; orchestrator delegates only.

## Escalation rules (ask before doing)
- Ask before adding any dependency not in "Tech stack" (esp. tenacity)
- Ask before adding files/dirs beyond "Architecture"
- Ask before changing the Phase 1 API contract or brand-config schema
- Ask before changing existing encoder / FAISS-build function signatures
- Ask if any existing test starts failing
- Ask if a single executor pass would touch more than 6 files
- Ask before any network call to a REAL brand endpoint in tests (tests must mock HTTP)

## Hard rules
- Never set `ANTHROPIC_API_KEY` (Max plan)
- Ingestion CALLS existing encoders + FAISS build logic — does not reimplement or modify them
- A brand with zero interaction data must still produce a queryable (cold-start) index
- Tests never hit live brand endpoints — Shopify path is tested against mocked responses
- API keys never written into yaml (only the env-var NAME)
- All PRs DRAFT; human merges
- Run full test suite after every executor pass
- Update PROJECT_MEMORY.md at phase end

## Budget
- Soft target: 1 CC session
- Hard cap: stop/escalate after 20 executor invocations
- `/cost` at midpoint

## Success criteria (verify ALL before done)
- `python scripts/ingest_catalog.py --source csv --path tests/fixtures/snitch_catalog.csv --brand snitch`
  produces `brands/snitch.yaml` + namespaced index + embeddings
- The Phase 1 API immediately serves `snitch`: `/v1/snitch/item/{id}/similar` returns
  results with NO interaction data ingested (cold-start works on a fresh catalog)
- Shopify path works against a mocked paginated `/products.json` response in tests
- Bad catalog rows (missing required column) rejected with a clear, actionable error
- Image download is resumable (re-run skips fetched images) and writes a failure manifest
- `ingest_interactions.py` produces user sequences; brand then serves personalized
  `/recommend` for an ingested user
- CLIENT_ONBOARDING.md walks a client data engineer through the full path end-to-end
- All prior passing tests still pass; new ingest/schema/image tests pass
- PROJECT_MEMORY.md updated with ingestion flow + CSV schema + onboarding summary

## Build order (orchestrator may adjust)
1. `app/ingestion/schema.py` — pydantic catalog + interaction row models (+ tests)
2. `sources.py` — CsvSource first, then ShopifySource (paginated, mocked in tests)
3. `images.py` — resumable download + retry + failure manifest (mocked http tests)
4. `pipeline.py` — wire fetch→embed (existing encoders)→index (existing build)→write yaml
5. `ingest_catalog.py` CLI + e2e fixture ingest → confirm Phase 1 API serves the brand
6. `ingest_interactions.py` + confirm personalized recommend works for ingested user
7. CLIENT_ONBOARDING.md
8. Update PROJECT_MEMORY.md
