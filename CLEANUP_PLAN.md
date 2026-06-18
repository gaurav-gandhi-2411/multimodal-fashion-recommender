# Cleanup Plan — Review AFTER Demo

> Generated 2026-06-19. **DO NOT execute before the demo.**
> The live Cloud Run service depends on files marked LOAD-BEARING.
> Anything marked SAFE has been confirmed not referenced by the live API or deploy pipeline.

---

## SAFE TO REMOVE

| Path | Size (approx) | Reason |
|------|---------------|--------|
| `data/h-and-m-personalized-fashion-recommendations/` | ~30 GB | H&M brand excluded from Cloud Run (`BRANDS_ENABLED=snitch,fashor,powerlook`). Raw training data not shipped in Docker. |
| `data/processed/` | ~156 MB | H&M FAISS indices + embeddings. Not synced to GCS; H&M brand not active in Cloud Run. |
| `spaces/` | ~11 MB | HuggingFace Spaces app — duplicates `app/streamlit_app.py` logic. Spaces deployment has been superseded by the Cloud Run API + this Next.js demo. Check if HF Space is still live first. |
| `brands/_snitch_rerank_backup.json` | ~1 KB | Untracked JSON backup of Snitch rerank config (superseded by `brands/snitch.yaml`). |
| `reports/similarity_eval_diversity.html` | ~2 MB | Ablation eval output HTML. Not loaded by API or CI. Untracked. |
| `reports/similarity_eval_occasion.html` | ~2 MB | Same as above. Untracked. |
| `*.log` files at repo root (`baselines.log`, `comparison.log`, `comparison_16ep.log`, `training.log`, `training_16ep.log`, `upload_artifacts.log`) | ~2 MB | One-time experiment output logs. Not referenced anywhere. |
| `C:Usersgaurapytest_output.txt` (repo root) | <1 KB | Windows path artifact from a pytest run. Untracked junk. |
| `notebooks/` (if exists) | varies | Exploration notebooks; not part of any pipeline. |
| `.pytest_cache/`, `.ruff_cache/` | varies | Local build caches; regenerated automatically. |
| `scripts/find_ab_items.py` | <1 KB | One-time A/B demo item discovery script. Not called by deploy or ingest. |
| `scripts/ablate_rerank.py` | <1 KB | One-time ablation harness. Phase 7 findings are locked in PROJECT_MEMORY.md. |
| `scripts/gen_catalog_json.py` | <1 KB | One-time catalog JSON generator for the Vercel demo. Already run; output in `demo/public/catalog/`. |

---

## DO NOT TOUCH — LOAD-BEARING

| Path | Why |
|------|-----|
| `app/` (entire directory) | FastAPI app. Cloud Run CMD = `uvicorn app.api.main:app`. Every module here is imported at startup. |
| `brands/snitch.yaml`, `brands/fashor.yaml`, `brands/powerlook.yaml` | Loaded at startup by `BrandRegistry`. Drive `BRANDS_ENABLED` filter and all GCS sync paths. |
| `brands/h_and_m.yaml` | Exists in image but filtered by `BRANDS_ENABLED`. Remove only after confirming H&M is permanently retired from all envs. |
| `src/` (entire directory) | Core ML modules (encoders, models, retrieval) imported by `app/` at inference time. Copied into Docker. |
| `data/snitch/`, `data/fashor/`, `data/powerlook/` | `items.parquet` + `transactions/` per brand. Synced from GCS at startup. |
| `indices/snitch/`, `indices/fashor/`, `indices/powerlook/` | FAISS retrieval indices + visual index + article_ids.pkl. Synced from GCS at startup. Without these, all recommendation endpoints fail. |
| `checkpoints/best.pt` | TwoTowerModel weights. Synced from GCS. Without this, user tower inference fails. |
| `infra/Dockerfile.cpu` | Used by `deploy.yml` (line: `-f infra/Dockerfile.cpu`). |
| `Dockerfile` | Used for local GPU dev. Not the Cloud Run deploy image but keep for GPU eval. |
| `pyproject.toml`, `uv.lock` | Dependency spec for Docker build. `uv sync --frozen --no-dev` in Dockerfile. |
| `config.yaml` | Copied into container. Referenced by training scripts; conservative: keep. |
| `.github/workflows/deploy.yml` | Manual deploy trigger. The only way to redeploy. |
| `demo/` | Next.js Vercel frontend. **Load-bearing for the demo itself.** `demo/public/catalog/*.json` is read by the Next.js API routes at runtime. |
| `scripts/ingest_catalog.py`, `scripts/ingest_interactions.py` | Needed to onboard new brands. Not called by deploy but essential for catalog rebuild. |
| `scripts/build_visual_index.py` | Rebuilds `indices/{brand}/visual.faiss`. Needed if catalog changes. |
| `scripts/prepare_indian_catalogs.py`, `scripts/prep_snitch_catalog.py` | Catalog prep scripts. Needed for catalog refresh. |
| `tests/` | 200 tests. Run in CI. Keep all. |

---

## RECOMMENDED CLEANUP ORDER (post-demo)

1. Delete `data/h-and-m-personalized-fashion-recommendations/` (saves ~30 GB, biggest win)
2. Delete `data/processed/` (H&M artefacts, ~156 MB)
3. Remove `spaces/` if HF Space is no longer maintained
4. Clean untracked junk: `_snitch_rerank_backup.json`, `reports/`, `*.log` files, `pytest_output.txt`
5. Archive one-time scripts (`find_ab_items.py`, `ablate_rerank.py`, `gen_catalog_json.py`) into `scripts/archive/`
