# ADR-0001: Multi-Brand Memory Scaling Architecture

**Status:** Accepted (current architecture); Under Review (recommended path)
**Date:** 2026-06-20
**Deciders:** Engineering Lead

---

## 1. Context

This system is a multi-tenant fashion recommendation API serving B2B clients (Indian fashion retailers). Each brand is isolated by API key and receives its own retrieval stack: a trained two-tower model (256-dim user and item embeddings), a FAISS `IndexFlatIP` over item embeddings, a separate CLIP ViT-B/32 visual index (512-dim), and a user transaction history DataFrame loaded from Parquet splits.

At startup, the container performs:

1. GCS sync — download brand artifacts (catalog, FAISS indices, model checkpoint, transaction Parquets) to local disk
2. Per brand: `FaissRetriever.load()` deserializes the FAISS index from disk into RAM
3. Per brand: `retriever.index.reconstruct_n(0, n_total)` reconstructs the **entire** embedding matrix into a separate `np.ndarray` (`item_embeddings` on `BrandState`) for O(1) candidate scoring in `/complete` (`registry.py:145-146`)
4. Per brand: a second `FaissRetriever.load()` for the CLIP-512 visual index (`registry.py:151-163`)
5. Per brand: `pd.concat()` of train/val/test transaction Parquet splits into a single in-memory DataFrame (`registry.py:117-127`)
6. Per brand: `torch.load()` of `checkpoints/best.pt` (~19 MB per brand)

Every brand's full state lives inside a single `BrandRegistry` dict, in the process address space of a single Cloud Run instance. There is no lazy loading, no eviction, and no shared retrieval tier.

The `BRANDS_ENABLED` environment variable gates which brand YAMLs are loaded, making it possible to exclude a brand without a code change, but the exclusion requires a full container redeploy to take effect.

---

## 2. Decision

**We accept the single-instance, eager-load model for the current PoC/demo phase (≤ 4 brands).**

This is an explicit architectural decision, not an oversight. The tradeoffs were evaluated and the following constraints drove the choice:

- **Simplicity:** zero external infrastructure dependencies; the entire system is a single `uvicorn` process
- **Local dev parity:** `docker-compose up` reproduces the production topology without a managed vector DB or a sidecar retrieval server
- **Demo readiness:** brand data is GCS-synced at startup so Cloud Run cold-start produces a fully warm instance; the first request pays no data-load penalty
- **CLIP weights baked in:** the encoder is included in the Docker image layer (`Dockerfile.cpu`), eliminating a 350 MB download on every cold start

**Currently deployed brands:**

| Brand | Catalog items | Two-tower FAISS | Visual FAISS | Deployed |
|---|---|---|---|---|
| Snitch | 1,803 | 1.8 MB | 3.5 MB | Yes |
| Fashor | 3,618 | 3.6 MB | 6.4 MB | Yes |
| Powerlook | 906 | 0.9 MB | 1.5 MB | Yes |
| H&M | ~10,556 (sampled) | ~20 MB est. | Not built | **No — excluded via `BRANDS_ENABLED`** |

H&M is present in `brands/h_and_m.yaml` with a trained index but is excluded from the live deployment. Its transaction history (2M rows subsampled from 31M, loaded via `pd.concat` at `registry.py:117-127`) pushes the 4 GiB Cloud Run instance over budget.

---

## 3. Consequences

### Memory model

Each `BrandState` holds simultaneously in RAM:

| Component | Powerlook (906 items) | Fashor (3,618 items) | H&M (10,556 items) |
|---|---|---|---|
| Two-tower FAISS `IndexFlatIP` (256-d float32) | ~0.9 MB | ~3.6 MB | ~10 MB |
| `item_embeddings` ndarray (duplicate at `registry.py:146`) | ~0.9 MB | ~3.6 MB | ~10 MB |
| Visual FAISS `IndexFlatIP` (512-d float32) | ~1.9 MB | ~7.4 MB | ~21 MB |
| Catalog DataFrame + `art_map` dict | ~5–20 MB | ~20–80 MB | ~60–200 MB |
| Transaction history DataFrame | ~5 MB (synthetic) | ~5 MB (synthetic) | ~500–1,500 MB (2M rows) |
| `TwoTowerModel` state dict | ~19 MB | ~19 MB | ~19 MB |
| **Subtotal** | **~30–50 MB** | **~60–115 MB** | **~620–1,760 MB** |

The `reconstruct_n` call at `registry.py:146` is the principal multiplier: it creates a third in-process copy of the embedding matrix (FAISS `IndexFlatIP` already stores the vectors verbatim; `reconstruct_n` allocates a separate ndarray). This was done for O(1) lookup in the `/complete` hot path but is the most expensive per-brand allocation.

### Scaling math

| Brands | Estimated peak RAM | Cloud Run tier needed | Est. monthly cost |
|---|---|---|---|
| 3 (current, live) | ~250–400 MB | 1 vCPU / 4 GiB | ~$15–40 |
| 4 (+ H&M) | ~1.5–2.5 GB | 2 vCPU / 8 GiB (tight) | ~$50–120 |
| 10 brands (mixed sizes) | ~8–15 GB | 4 vCPU / 16 GiB | ~$200–500 |
| 50 brands | ~40–75 GB | Not achievable on a single Cloud Run instance | N/A |

### Boot time

Current cold-start for 3 brands: **15–35 seconds**. At 10 brands: ~60–120 s. Cloud Run's default start timeout is 240 s; 30+ brands would breach it.

### Add-brand latency

Today, adding a brand requires:
1. Upload artifacts to GCS (~2 min)
2. Add YAML + commit + push (~2 min)
3. Docker build via Cloud Build (~30 min on E2_HIGHCPU_8)
4. Cloud Run deploy (~5–10 min)
5. Startup (~30 s)

**Minimum lead time from "data ready" to "brand live": ~45 minutes, dominated by container rebuild.**

---

## 4. Alternatives Considered

| Option | Brands supported | Add-brand latency | RAM scaling | Complexity |
|---|---|---|---|---|
| **A. Lazy loading + LRU eviction** (current process) | 20–50 (with per-brand cold-start) | ~5 min (no rebuild) | Fixed ceiling at instance size | Low — ~200 lines in `registry.py` |
| **B. Shared FAISS shard server** (retrieval sidecar) | 10–30 | ~2 min (ingest + index) | Scales independently | Medium |
| **C. Shared model weights** (one checkpoint across brands) | Orthogonal | No change | Saves N×19 MB | Medium (retraining) |
| **D. Managed vector DB** (Qdrant Cloud / Pinecone / Weaviate) | 50–1,000+ | ~2 min (ingest API) | Zero in-process RAM for vectors | Low-Medium (managed) |
| **E. Per-brand Cloud Run instances** (gateway routing) | Unlimited | ~10 min (per-brand build) | Each instance sized independently | High (ops scales with N) |

### Option A — Lazy loading + LRU eviction

Pure code change to `registry.py` and `main.py`. `load_registry()` becomes a `LazyBrandRegistry` that wraps `_load_brand()` behind a lock, records last-access time, and evicts the LRU brand when `psutil.Process().memory_info().rss` crosses a threshold.

Risk: first request to an evicted brand pays full GCS-sync + FAISS-reconstruct cold start (10–30 s). Acceptable for B2B SaaS where brands have predictable traffic patterns. Not acceptable for consumer products with random brand access.

### Option B — Shared FAISS shard server

Decompose retrieval from the API tier. FAISS indices move to a dedicated persistent process (custom gRPC server, or self-hosted Qdrant). The API container holds only the `TwoTowerModel` for query-time embedding and calls the retrieval service over gRPC.

The `FaissRetriever` class already exposes a clean `search(query_emb, k)` interface; swapping the implementation is a thin network adapter. Latency addition: ~5–15 ms on Cloud Run VPC (same region).

### Option D — Managed vector DB (production target)

The canonical production architecture. FAISS `IndexFlatIP` is semantically equivalent to a cosine-similarity flat index in any managed vector DB. The `FaissRetriever.save()` / `FaissRetriever.load()` calls are replaced by SDK calls to a Qdrant collection (one collection per brand, or one collection with a `brand_id` payload filter for multi-tenant isolation).

API startup becomes: authenticate → verify collections exist → done. No GCS sync, no FAISS reconstruct, no Parquet concat. Cold-start drops to ~5 s.

The `reconstruct_n` allocation at `registry.py:146` disappears: Qdrant returns stored payload alongside vectors, eliminating the need for a separate `item_embeddings` matrix and `faiss_row_to_aid` lookup table.

Add-brand flow: ingest script → vectors upserted → YAML committed → API reload (no container rebuild, no 30-min build).

Cost: Qdrant Cloud starts at ~$25/month for a 1M-vector cluster. At 50 brands × 5,000 items average = 500k vectors across two-tower (256-d) and CLIP (512-d) indices, well within the entry tier.

---

## 5. Recommended Migration Path

### Phase 1 — Now (≤ 4 brands, demo/PoC)
**Architecture:** Current single-instance eager-load. No action needed. Use `BRANDS_ENABLED` to gate H&M out until its transaction history is trimmed or an 8 GiB instance is justified by a paying customer.

### Phase 2 — Near-term (5–15 brands)
**Architecture:** Lazy loading + LRU eviction (Option A) on existing Cloud Run deployment.

Implementation: refactor `BrandRegistry` to `LazyBrandRegistry`. Set memory threshold at 75% of instance RSS. Upgrade to 2 vCPU / 8 GiB when a brand with real traffic is added. Communicate per-brand cold-start SLA (P99 first-request < 30 s) to clients.

No external infrastructure changes required.

### Phase 3 — Production (15–50 brands)
**Architecture:** Qdrant Cloud (or self-hosted Qdrant on GKE) as shared retrieval tier; API container is stateless.

Steps:
1. Replace `FaissRetriever` with a `QdrantRetriever` implementing the same `search()` interface
2. Migrate `scripts/build_index.py` to upsert to Qdrant instead of writing FAISS files to disk
3. Remove `reconstruct_n` allocation and `item_embeddings` from `BrandState`; use Qdrant payload retrieval in `/complete`
4. Add-brand latency drops to: ingest + YAML commit + API reload (no container rebuild)

### Phase 4 — Enterprise (50+ brands with SLA tiers)
**Architecture:** Per-brand Cloud Run instances behind a routing gateway (Cloud Endpoints or Cloudflare Workers route by `X-Brand-Id`), with Qdrant as shared retrieval tier.

Rationale: brand isolation becomes a contractual requirement for enterprise clients (data residency, SLA independence). The per-brand instance model achieves full data-plane isolation without a multi-tenant vector DB.

---

## 6. Open Questions

1. **Transaction history at scale:** The `pd.concat` approach at `registry.py:117-127` is not viable for brands with real transaction volumes (H&M: 31M rows, Fashor/Snitch: synthetic). For Phase 3, user history should move to a feature store (Redis sorted sets by user_id, or BigQuery for batch inference).

2. **Model per-brand vs. shared:** The current `checkpoints/best.pt` is trained on H&M data; Snitch/Fashor/Powerlook use it as a zero-shot retriever. Brand-specific fine-tuning improves quality but adds N × 19 MB checkpoint cost. Evaluate cross-brand quality loss before committing to a shared model.

3. **Runtime brand registration:** `BRANDS_ENABLED` is a coarse startup gate, not a management plane. A production multi-brand API needs a `/admin/brands` endpoint. Out of scope until Phase 3.

---

## Appendix: Key Code Locations

| Concern | File | Lines |
|---|---|---|
| Per-brand eager load | `app/brands/registry.py` | `_load_brand()`, 103–178 |
| `reconstruct_n` allocation | `app/brands/registry.py` | 145–146 |
| Visual index load | `app/brands/registry.py` | 151–163 |
| Transaction history concat | `app/brands/registry.py` | 117–127 |
| GCS sync | `app/storage.py` | `sync_brand_assets()` |
| Startup sequence | `app/api/main.py` | `lifespan()`, 22–39 |
| FAISS retriever interface | `src/retrieval/faiss_index.py` | `FaissRetriever` |
| Brand YAML schema | `app/brands/registry.py` | `BrandConfig`, 43–62 |
