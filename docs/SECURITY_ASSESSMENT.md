# Security Assessment

**Scope:** buyer technical diligence — is the system defensible as-is, not "is it bulletproof."
**Method:** direct code inspection (file:line citations below) plus a scoped dependency
vulnerability scan (`pip-audit` against the locked Python dependencies, `npm audit` against
the demo's locked Node dependencies). No penetration test was run against the live deployment.

**Bottom line:** no Critical findings. One High-severity, concretely reachable issue was found
in a transitive dependency (Starlette) and has been fixed in this change (dependency bump, no
application code touched). Everything else is Low or Informational, already substantially
mitigated by existing controls.

---

## 1. Authentication / authorization (per-brand API keys)

**Current state:** every brand-scoped endpoint depends on `require_brand`
(`app/api/auth.py:8-29`), which resolves the brand from the URL path, looks up that brand's
specific `BrandState` in the in-memory registry, and compares `X-Api-Key` against
`BrandState.api_key` — a value that differs per brand. Each brand's key is provisioned as an
independent Google Secret Manager secret in `.github/workflows/deploy.yml` (5 distinct secrets
for 5 brands) and delivered to the container via `--set-secrets`, never as plaintext
`--set-env-vars`.

**Cross-tenant check:** Brand A's key cannot authenticate against Brand B's URL — the
comparison is against a different `BrandState.api_key` value entirely, not a shared secret or a
prefix/pattern check. Verified by direct read of the dependency, not inferred from
documentation.

**The semi-public sandbox key (H&M):** this key is intentionally distributed for the public
demo. It authenticates only the H&M `BrandState` — it grants no access to any other brand's
data or keys. The residual concern is cost, not data exposure — see §2.

**Risk: Low.** Correct as implemented.

## 2. Rate limiting

**Current state:** `slowapi` limiter keyed on `f"{ip}:{brand}"` (`app/api/rate_limit.py:8-12`)
— per-(caller IP, brand), not a single global counter. There is no additional
cross-caller/global ceiling at the application layer.

**Mitigation already in place:** Cloud Run itself caps the service at
`--max-instances 3 --concurrency 80` (`.github/workflows/deploy.yml:119-124`), which bounds
total concurrent load regardless of how many distinct IPs an attacker spreads across — a
practical ceiling on cost-DoS even without a global app-level limiter.

**Risk: Low.** A dedicated attacker rotating source IPs could still multiply their own
per-IP allowance, but the Cloud Run instance/concurrency ceiling bounds the blast radius to a
fixed, budgeted cost — not an open-ended bill. Sufficient for the current stage; a global
token-bucket would be the next increment if traffic volume grows enough to justify it.

## 3. Input validation (image upload, injection surface)

**Current state:**
- `visual_search` (`app/api/routes.py:671-682`) enforces a 10MB cap on the uploaded image
  after `await image.read()` (`app/api/routes.py:712-721`); oversized or corrupt images are
  converted to `HTTPException(400)` (`app/api/routes.py:726-731`), not a raw 500.
- Image decoding goes through `Image.open(...).convert("RGB")` (`app/visual.py:77`). No
  `Image.MAX_IMAGE_PIXELS` override exists anywhere in the repo (confirmed by grep), so
  Pillow's own decompression-bomb guard (raises above ~89M decoded pixels) is active and
  unmodified.
- **No SQL database exists anywhere in this project** (confirmed by grep across `app/`) — the
  catalog/config/index data path is parquet + FAISS + YAML, not a query layer, so SQL
  injection is not an applicable class of vulnerability here.

**Risk: Low.** Standard boundary validation is present; no injection surface found because
there is no query interpreter in the request path.

## 4. Secret handling (logs, repo, client)

**Current state:**
- `app/api/logging_config.py` (read in full) never references `X-Api-Key` or raw request
  bodies in any structured log call.
- `infra/Dockerfile.cpu` (read in full) bakes in only public FashionCLIP model weights at
  build time; brand assets, API keys, and the Groq key are synced/injected at container
  runtime via `app/storage.py` and Secret Manager `--set-secrets`, never baked into the image
  or committed to the repo.
- No personal or service credentials found in the repository via grep (separate from the
  professional-cleanup pass, which checked for personal *names*, not secrets).

**Risk: Low.** No secret leakage path found in logs, image layers, or git history.

## 5. The Vercel demo proxy

**Current state:** all 5 proxy routes (`demo/app/api/{personalized,visual-search,style-search,
similar,complete}/route.ts`, each read in full) read their backend API key from
`process.env.FASHION_API_KEY_*!` — a server-only environment variable in a Next.js Route
Handler, never sent to the browser. Backend error responses are relayed via
`formatBackendError`, which forwards only the backend's error text, never the key itself.

**Risk: Low.** The proxy does what its own internal `QUICKSTART.md` claims it does — keys stay
server-side.

## 6. Dependency vulnerabilities

This is the one area where the original research pass (manual version comparison against
`uv.lock`) was explicitly flagged as incomplete — it couldn't tell you which pins actually carry
known CVEs. That gap is closed in this change: both ecosystems were run through a live,
network-backed scanner.

**Python — `pip-audit`, scoped to the actual locked runtime dependencies (`uv export
--frozen --no-hashes`, GPU-specific `torch`/`torchvision` build excluded — no PyPI-audited
CVE feed covers CUDA wheel builds, and they carry no HTTP-facing surface):**

| Package | Locked version | Advisory | CVSS | Fixed in | Reachable in this app? |
|---|---|---|---|---|---|
| starlette | 1.2.1 | PYSEC-2026-249 (CVE-2026-54283) | 7.5 High | 1.3.1 | **Yes — see below.** |
| starlette | 1.2.1 | PYSEC-2026-248 (CVE-2026-54282) | 5.3 Medium | 1.3.0 | Host-spoofing via malformed path; app does not trust `request.url.hostname` for authz decisions anywhere — low practical impact, fixed as a side effect of the bump below. |
| pydantic-settings | 2.14.1 | GHSA-4xgf-cpjx-pc3j | 5.3 Medium | 2.14.2 | **No** — requires `BaseSettings` with a symlinked `secrets_dir`. Grepped `app/`: this project never uses `BaseSettings`/`secrets_dir` at all (transitive dependency, unused feature). Not applicable to this deployment. |

**PYSEC-2026-249 detail (the one that mattered):** Starlette's `request.form()` enforces its
size/field limits for `multipart/form-data` but silently ignores them for
`application/x-www-form-urlencoded` — an unbounded parse, CVSS 7.5. This app has one endpoint
that accepts `UploadFile = File(...)` (`visual_search`,
`app/api/routes.py:671-676`), which routes through Starlette's `request.form()`. FastAPI/
Starlette parse the request body **before** the `require_brand` dependency or the `slowapi`
rate-limit decorator execute — so this was reachable **without a valid API key and without
being rate-limited**, by sending the endpoint a request with
`Content-Type: application/x-www-form-urlencoded` and an oversized body instead of the expected
multipart image. This is a real, unauthenticated memory/CPU exhaustion vector, not a
theoretical CVSS number.

**Fix shipped in this change:** `uv lock --upgrade-package starlette` → 1.2.1 → 1.3.1 (fixes
both advisories; no application code changed). Verified: full test suite re-run post-bump,
332 passed / 1 pre-existing-and-unrelated failure (a two-tower training dataset key-mismatch
test, confirmed to fail identically on `main` before this change — not caused by the bump, not
part of this system's HTTP-facing behavior).

**Node — `npm audit` against the demo's locked dependencies:**

`next@15.4.11` has open advisories (1 High summary bucket, aggregating several CVEs — DoS via
Image Optimizer, middleware/proxy bypass classes, cache poisoning; 1 Moderate via a transitive
`postcss`). Reachability check against this specific app:
- No `middleware.ts` exists in `demo/` (grep confirmed) — the middleware-bypass and
  cache-poisoning classes don't apply; there is no middleware to bypass.
- No `i18n` config exists — the i18n-routing bypass class doesn't apply.
- `next/image` (the component) is not used anywhere in application code (only the
  auto-generated `next-env.d.ts` type reference) — but `next.config.ts:4-17` does configure
  `images.remotePatterns`, so the built-in `/_next/image` optimizer endpoint is present and
  configured, keeping the Image-Optimizer-DoS advisory technically live.
- **Deployment context matters here:** this demo runs on Vercel, not a self-hosted
  `next start`/standalone container. The GHSA advisory titles explicitly scope the
  disk-cache-exhaustion and several DoS variants to *self-hosted* deployments using Next's own
  local file-system image cache — Vercel's platform serves image optimization through its own
  infrastructure, not the vulnerable local-disk code path described in the advisory.
- No stable fixed release exists yet in the 15.x line (checked `npm view next versions`: latest
  stable is `15.5.20`, and `npm audit fix --dry-run` still reports the same advisories against
  it) — a full fix requires Next 16, currently only in preview (`16.3.0-preview.5`), not a
  released stable version.

**Risk: Low, not fixed in this change.** Reachable surface is narrowed by the deployment
platform and the absence of middleware/i18n; no released stable version exists that would
resolve it without an unreleased major-version jump. Recommendation: track Next 16 stable and
upgrade when it ships, rather than adopt a preview build in a customer-facing demo.

## 7. Data exposure / multi-tenant isolation

**Current state:** `BrandRegistry` (`app/brands/registry.py`) holds one independent
`BrandState` per brand — `catalog`, `retriever` (FAISS index), `art_map`,
`faiss_aid_to_row`/`faiss_row_to_aid`, `item_embeddings`, `visual_retriever`, `color_index`,
`attributes`, `user_history`, and `api_key` are all separate Python objects per brand, built
fresh per YAML config file at startup (`_load_brand()`, `app/brands/registry.py:117-210`).
There is no shared/global FAISS index, no shared catalog table, and no cross-brand key that
could accidentally resolve into another brand's data.

**Risk: Low.** Genuine per-brand memory isolation, confirmed by reading the registry
construction path, not inferred from naming conventions.

---

## What this assessment does not cover

- No live penetration test or fuzzing pass against the deployed staging environment.
- No infrastructure-level review (IAM bindings, Cloud Run service account scope,
  VPC/firewall config) beyond what's visible in the deploy workflow.
- Dependency scan is a point-in-time snapshot (today's advisory database) — re-run
  `pip-audit`/`npm audit` on a cadence, not just once.

## Summary table

| Area | Risk | Action taken |
|---|---|---|
| Per-brand auth/authz | Low | None needed |
| Rate limiting | Low | None needed |
| Input validation | Low | None needed |
| Secret handling | Low | None needed |
| Vercel proxy | Low | None needed |
| Python dependency CVEs | **High → fixed** | `starlette` 1.2.1 → 1.3.1 (this PR) |
| Node dependency CVEs | Low | Documented; revisit on Next 16 stable |
| Multi-tenant isolation | Low | None needed |
