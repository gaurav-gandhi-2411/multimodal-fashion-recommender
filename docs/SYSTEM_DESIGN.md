# System Design — Technical Diligence One-Pager

**Purpose:** an honest answer to "how does this run in production, and where does it break."
Written for a technical buyer, not a portfolio reader — strengths and the one known scaling
ceiling are both stated plainly.

---

## Architecture at a glance

FastAPI service on Cloud Run (`asia-south1`). One process serves every brand. At startup,
`BrandRegistry.load_registry()` reads every `brands/*.yaml`, and for each one builds a fully
independent `BrandState` — its own catalog (parquet), FAISS retrieval index, visual
(FashionCLIP) index, color/attribute indices, and API key — all held in the same process's
memory (`app/brands/registry.py`). Recommendation/similarity scoring runs a shared two-tower
model (`checkpoints/best.pt`) against each brand's own embeddings; visual search runs a shared
FashionCLIP encoder against each brand's own visual index. No brand's data is ever shared with
another's at the object level.

## 1. Reproducibility

Every eval number cited in this project's documentation can be regenerated from a clean
checkout: eval scripts live in `scripts/*eval*.py` (e.g.
`scripts/07_coldstart_stratified_eval.py`, `scripts/calibrate_match_confidence.py`), each seeded
(`seed=42` convention), reading directly from the same parquet/FAISS artifacts the live service
uses — not a separate offline copy. There is no eval number in this project's history that
required hand-tuning or cherry-picked sampling to reproduce; every reported percentage has an
`n=` and a raw-vs-reranked comparison alongside it (see `reports/*.html`).

**Gap, stated honestly:** eval scripts are run manually, not wired into a CI gate
(`eval.yml` does not exist yet). A regression in reranking quality would not fail a build; it
would only be caught by someone re-running the eval by hand. This is the single most concrete,
fixable ML-Ops gap in the project — flagged here, not fixed in this pass (adding CI eval-gating
is new infra, out of scope for an assess-only pass).

## 2. Model versioning

**Gap, stated honestly:** `checkpoints/best.pt` (the two-tower model) is gitignored — it is
never committed, and carries no embedded version identifier. Loading it directly
(`torch.load(..., weights_only=False)`) shows its dict keys are `epoch`, `model_state_dict`,
`optimiser_state_dict`, `metrics`, `config` — there is no hash, timestamp, or semantic version
inside the artifact itself. Today, "which model is live" is answered by inference from deploy
history and PROJECT_MEMORY.md's narrative log, not from the artifact.

**What already substitutes for it:** the "two-tower byte-identical proof" performed before every
deploy that could plausibly touch the serving path (see §3) — comparing `/recommend`,
`/similar`, `/complete` outputs byte-for-byte between the old and new revision via Cloud Run's
tagged-revision technique. This proves *behavioral* continuity even without a version string.
It does not answer "what training run produced this checkpoint" after the fact — that would
require an actual model registry (MLflow run ID embedded in the checkpoint, or a checkpoint
hash recorded at deploy time). Real follow-up, not built here: embed the MLflow run ID in the
checkpoint's `config` dict at training time.

## 3. Deploy safety — the Deploy Verification Standard

This is the strongest-evidenced part of the operational story: a documented, repeatedly-applied
standard (`PROJECT_MEMORY.md:296`, "binding — read before any serving-path dependency change"),
applied consistently across every deploy touching the serving path:

1. **Container-verify**: build `infra/Dockerfile.cpu` from the exact commit being deployed, run
   it locally with real brand data bind-mounted, hit the real endpoints inside that container
   — before touching Cloud Run at all. Rationale recorded in the standard itself: "local pass
   ≠ container pass" (dependency resolution genuinely differs between a dev machine and the
   Docker build).
2. **GCS preflight**: confirm every brand asset the new code path needs is actually present in
   GCS before deploying — this exact class of gap (data present locally, absent in the bucket
   the container syncs from) caused a real incident once and is now a standing checklist item.
3. **Trigger via `workflow_dispatch`** (`.github/workflows/deploy.yml`), never a push-triggered
   auto-deploy — a deliberate manual gate.
4. **merged==live proof**: after deploy, `gh run view --json headSha` is compared against the
   deployed revision's source commit — an exact SHA match, not "should be the same." Caught a
   real docs-only gap once (Phase 9); zero-gap deploys are now the norm and are checked every
   time, not assumed.
5. **Regression-check all live brands**, not just the one being changed.
6. **Two-tower byte-identical proof**: Cloud Run's tagged-revision technique
   (`gcloud run services update-traffic --set-tags <name>=<old-revision> --to-latest`) lets both
   the old and new revision be queried side-by-side on the same live traffic split, diffed for
   byte-identical output, then the tag removed. Used before every deploy that could plausibly
   touch scoring — most recently confirmed byte-identical across snitch/fashor/powerlook after
   the Virgio onboarding deploy.

This is a real, repeatable process — not an aspirational description. It has caught real bugs
before they reached 100% traffic (the "5th boot blocker": a raw-YAML read that skipped a
pydantic default, causing `FileNotFoundError` at `torch.load` — caught by container-verify, not
in production).

## 4. Monitoring / observability

Prometheus metrics exposed at `/metrics` (`app/api/metrics.py`): `REQUEST_COUNT`,
`REQUEST_LATENCY`, `LLM_COST_USD`, `LLM_CALLS`, `LLM_CALL_DURATION`, `LLM_TOKENS_TOTAL`,
`EXPLANATION_CACHE_HITS`, `EXPLANATION_CACHE_MISSES` — request volume, latency distribution, and
LLM cost/usage are all instrumented per-call. Structured logging via `structlog`
(`app/api/logging_config.py`) never logs raw request bodies or API key values, so logs are safe
to ship to an external aggregator without a redaction pass.

**Honest gap:** no distributed tracing (OpenTelemetry/Langfuse) and no external alerting
(Sentry) are wired up yet — an incident would be debugged from Cloud Run's own log viewer and
the 8 Prometheus series, not a trace waterfall. Sufficient to diagnose "which endpoint, how
slow, how much LLM cost" — not sufficient to diagnose a subtle cross-request causal chain
without manual log correlation by `request_id`.

## 5. Scaling limits — the real ceiling, stated honestly

**The architecture:** every brand's full object graph loads into the same process's RAM at
container startup. This is a deliberate, simple design — not an oversight — and it has been
adequate through 5 brands.

**What the raw numbers say:** current per-brand on-disk footprint is small — roughly 4–15 MB
per brand today (largest, `fashor`: catalog 1.1MB + embeddings 3.6MB + FAISS index 3.6MB +
visual index 6.5MB ≈ 15MB total). FashionCLIP itself (shared across all brands, loaded once)
adds a measured 589.8MB resident memory — 15% of the current 4Gi Cloud Run allocation
(`.github/workflows/deploy.yml:121`, PROJECT_MEMORY.md's own Phase 9 measurement). By raw
memory arithmetic alone, there is headroom for several dozen more brands before hitting the 4Gi
ceiling.

**The real constraint is not memory — it's blast radius.** A single-process, load-everything-
at-startup design means one bad brand config prevents the *entire fleet* from booting, not just
that one brand. This has already happened multiple times at only 3–5 brands: Phase 7's boot
blockers (a missing GCS binding, then a 5th blocker where a raw-YAML read skipped a pydantic
default and crashed `torch.load`), and a brand config that sat dead in the repo since Phase 7
specifically because it could not boot without crashing the whole service
(`BRANDS_ENABLED` filter introduced as the mitigation). Three-plus real incidents at 5 brands is
the actual signal — not the RAM math.

**The honest path past it** (not built, explicitly deferred — this is "Tier 4" work,
PROJECT_MEMORY.md's own already-identified next tier): either (a) lazy/on-demand per-brand
loading instead of eager-all-at-startup, so one brand's failure doesn't block others from
serving, (b) horizontal sharding — route each brand (or brand group) to its own Cloud Run
service/revision, isolating failure domains entirely, or (c) an external vector database
(e.g. a managed FAISS/pgvector service) so brand data isn't in-process memory at all. None of
these are built. This is a known, named gap — not a surprise a buyer would find later.

## 6. Rollback capability

Cloud Run revisions are immutable and traffic-tagged; the tagged-revision technique used for
the byte-identical proof (§3) is the same mechanism that makes rollback a traffic-routing change
(`gcloud run services update-traffic`), not a redeploy — the previous revision is still running
and can receive 100% of traffic again in seconds if a new deploy misbehaves. This has been
exercised as part of routine verification (tag old revision, compare, remove tag) though not yet
under an actual incident.

## 7. Eval/serve-path divergence guard

A specific, named tech-debt class in this project's own history: reranking logic once existed
in a form where the eval script and the live `/recommend` route could compute slightly different
things, making an eval number not actually represent what a real caller receives. The fix
pattern applied repeatedly since: derive both the eval path and the serving path from the *same*
underlying function (e.g. `brand_asset_paths()` extracted so the runtime GCS sync and the CLI
preflight tool can never drift apart; diversity reranking computed inside the shared `rerank()`
call so eval and route can't diverge). This is a design discipline, not a single fix — every
new rerank feature is checked against "does the eval measure the exact code path a live request
executes," and multiple PRs explicitly cite closing this exact divergence class.

---

## Summary — what a buyer should take away

| Dimension | State |
|---|---|
| Reproducibility | Strong — every number re-runnable from clean checkout; not yet CI-gated |
| Model versioning | Real gap — no artifact-embedded version; behavioral proof substitutes, provenance doesn't |
| Deploy safety | Strong — documented, repeatedly-applied, catches real bugs before 100% traffic |
| Monitoring | Adequate for current scale — no tracing/external alerting yet |
| Scaling ceiling | Known, honestly named — blast-radius risk at the process level, not a RAM limit; path past it is deferred, not hidden |
| Rollback | Strong — traffic-tag based, seconds not minutes |
| Eval/serve divergence | Actively guarded against as a design discipline |
