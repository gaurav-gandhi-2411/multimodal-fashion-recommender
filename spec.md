# Project Spec: multimodal-fashion-recommender — Phase 4 (Deploy + Cost + Caching)

## Goal
Turn the working local product into a deployed one a client can integrate against, and
attach the single number that drives every pricing conversation: cost per 1,000
recommendations (USD and INR). After this phase there is a live Cloud Run endpoint, each
call's cost is logged and exposed, and LLM explanations are cached so cost stays bounded.
This is the phase that makes "here's our pricing" a sentence you can say with a real
number behind it.

## Current state (existing project — DO NOT break)
Phases 0–3 complete on main:
- Validated model (3.06× pop / 2.12× co-purchase), multi-tenant API, ingestion pipeline,
  3 Indian brands (Snitch/Fashor/Powerlook) with verified live PDP URLs, white-label demo
- API: `/v1/{brand}/recommend`, `/v1/{brand}/item/{id}/similar`, `/health`, `/metrics`,
  per-brand `X-Api-Key`, `request_id`, structlog, basic Prometheus counters, Docker image
- 117 tests pass (1 known pre-existing failure)
- Known issue from Phase 1: Docker image is 2.9 GB because uv.lock pulls CUDA runtime
  into the CPU container — this phase fixes it (CPU-only lockfile) since it directly hits
  Cloud Run cold-start and cost.

Load-bearing — do NOT change without escalating:
- Model / encoders / eval harness / FAISS build logic
- API contract (routes, schemas incl. `request_id`, auth)
- Ingestion pipeline + brand-config schema
- Existing tests

## Scope

### In scope (this iteration)
**Deploy (the point of the phase):**
- CPU-only Docker image: separate CPU-resolved dependency set (e.g. `uv.lock` CPU extra
  or a `requirements-cpu` path) so the Cloud Run image drops the unused CUDA runtime.
  Target a materially smaller image (report before/after size).
- `Dockerfile` confirmed to run on Cloud Run (PORT env, non-root, healthcheck on `/health`).
- GitHub Actions `deploy.yml`: build → push to Artifact Registry → deploy to Cloud Run,
  triggered manually (workflow_dispatch) — NOT auto-deploy on push.
- Brand indices/embeddings loaded from GCS at startup (Cloud Run has no persistent disk);
  config points at GCS paths. Local dev still works from local paths.

**Cost transparency (the sales hook):**
- `app/pricing.py` — pricing constants: Groq $/1M tokens (input+output), Cloud Run
  $/vCPU-sec + $/GiB-sec, USD→INR rate (configurable constant). Source the Groq rate via
  a documented constant, not hardcoded magic.
- Per-call cost computed and logged as structured fields: `usd_cost`, `inr_cost`,
  `llm_tokens`, plus whether the explanation was a cache hit (cost 0 on hit).
- A small `/v1/{brand}/cost-summary` or a logged rollup that answers "cost per 1,000
  recommendations" — the deliverable number. Document how it's derived.

**Caching (so cost is bounded):**
- Explanation cache keyed by (brand + ranked item_ids hash + user-archetype/cold-start
  flag). Redis if available, with an in-process LRU fallback so local/dev works with no
  Redis. Cache hit → skip LLM call → cost 0 for that explanation.
- `docker compose` for local: API + Redis (+ optionally Prometheus) so the cache path is
  exercisable locally.

**Metrics (extend what exists, keep it light):**
- Prometheus histograms: `recommendation_latency_seconds`, `llm_call_duration_seconds`,
  `llm_tokens_total`, `explanation_cache_hits_total` / `..._misses_total`, all brand-labelled.

### Out of scope (escalate if you think it's needed; default NO this phase)
- Grafana dashboards, Sentry, OpenTelemetry/Jaeger tracing — defer to a later observability
  pass; metrics endpoint + structured cost logs are enough to sell on. (Escalate if you
  believe one is trivial to add and worth it.)
- A/B / champion-challenger, feedback loop, Postgres click logging (Phase 5 — likely
  deferred until a real client per the roadmap discussion)
- Any model change / retrain

## Tech stack
- Existing stack + redis (python `redis` client), Cloud Run, Artifact Registry, GCS
- google-cloud-storage for GCS index loading
- Escalate before adding anything beyond redis + google-cloud-storage

## Architecture
```
app/
  pricing.py            # cost constants + per-call cost computation
  cache.py              # explanation cache: Redis client + in-process LRU fallback
  storage.py            # GCS index/embedding loader (local-path fallback for dev)
infra/
  Dockerfile.cpu        # or modify existing Dockerfile to CPU-only resolution
  docker-compose.yml    # API + Redis (+ Prometheus) for local
.github/workflows/
  deploy.yml            # workflow_dispatch → build/push/deploy to Cloud Run
tests/
  test_pricing.py       # cost math correctness (tokens → usd → inr)
  test_cache.py         # hit/miss, key stability, LRU fallback when no Redis
```

## Verification commands
```yaml
- name: tests
  cmd: pytest -v
  required: true
- name: lint
  cmd: ruff check .
  required: true
- name: docker_build_cpu
  cmd: docker build -f infra/Dockerfile.cpu -t fashion-rec:cpu .
  required: true
- name: compose_up_smoke
  cmd: docker compose -f infra/docker-compose.yml up -d && sleep 15 && curl -f localhost:8000/health
  required: true
```

## Subagent usage rules
- `executor` writes/edits; `verifier` runs checks; orchestrator delegates only.

## Escalation rules (ask before doing)
- Ask before adding any dependency beyond redis + google-cloud-storage
- Ask before adding files/dirs beyond "Architecture"
- Ask before changing the API contract, ingestion pipeline, or any model/encoder code
- Ask before anything that would touch GCP project `aetherart-497918` — NEVER touch it;
  use a separate project/bucket for this product, confirm which before deploying
- Ask if any existing test fails
- Ask if an executor pass would touch more than 6 files
- Ask before hardcoding any credential or committing any GCP key (must use env/secret)
- Escalate the actual Cloud Run deploy step for human execution (don't auto-deploy)

## Hard rules
- Never set `ANTHROPIC_API_KEY` (Max plan)
- NEVER touch GCP project `aetherart-497918`
- No secrets/keys/service-account JSON committed to the repo — env vars / GitHub secrets only
- WORK ON A FEATURE BRANCH and open a DRAFT PR — do NOT commit directly to main
  (Phase 3 committed to main by mistake; restore the branch+PR gate this phase)
- Caching/cost must degrade gracefully: no Redis → LRU fallback; cost constants missing
  → log a clear warning, don't crash
- Run full test suite after every executor pass
- Update PROJECT_MEMORY.md at phase end

## Budget
- Soft target: 1 CC session
- Hard cap: stop/escalate after 20 executor invocations
- `/cost` at midpoint

## Success criteria (verify ALL before done)
- CPU-only image builds and is materially smaller than 2.9 GB (report the number)
- `docker compose up` starts API + Redis; `/health` returns OK
- Second identical explanation request is a cache hit (no LLM call, cost 0) — proven by test
- Per-call structured logs include `usd_cost`, `inr_cost`, `llm_tokens`, `cache_hit`
- The "cost per 1,000 recommendations" figure is derivable and documented (in
  PROJECT_MEMORY.md or a COST.md), with the assumptions stated
- `/metrics` exposes the new histograms + cache hit/miss counters, brand-labelled
- `deploy.yml` exists as workflow_dispatch (reviewed, NOT auto-run); GCS index loading
  works (tested against a bucket that is NOT in project aetherart-497918)
- All prior tests pass; new pricing + cache tests pass
- DRAFT PR opened from a feature branch (not committed to main)
- PROJECT_MEMORY.md updated

## Build order (orchestrator may adjust)
1. Feature branch off main
2. `app/pricing.py` + `test_pricing.py` (cost math)
3. `app/cache.py` + `test_cache.py` (Redis + LRU fallback, wired into explanation path)
4. New Prometheus histograms + cache counters
5. `app/storage.py` GCS loader (local-path fallback)
6. CPU-only Dockerfile + `docker-compose.yml`; build + compose smoke
7. `deploy.yml` (workflow_dispatch)
8. COST.md / PROJECT_MEMORY.md: cost-per-1000 derivation + assumptions
9. Open DRAFT PR
