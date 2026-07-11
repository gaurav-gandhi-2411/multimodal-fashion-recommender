# 0001 - One-command brand onboarding runbook (Tier 3)

## Context

Adding a new brand tenant to the live product is today 5 manual, incident-prone steps run
by us (never the client — Tier 4 self-serve is explicitly out of scope): catalog ingestion,
GCS asset upload, Secret Manager key creation, `BRANDS_ENABLED` + `deploy.yml` patch, and
triggering the Cloud Run redeploy. Three real incidents trace directly to this being manual:

- Phase 7: GCS index-dir sync bug, non-root permission bug, `BRANDS_ENABLED` comma-parsing
  bug, and a 5th boot blocker (`checkpoint_path` default skipped because `storage.py` read
  raw YAML instead of a validated `BrandConfig`) — four separate boot-crash classes, all
  found only once a real deploy was attempted.
- Phase 9: `uv.lock` resolved `transformers==5.10.2` for the Docker build while local testing
  ran against `4.57.6` — an 8-minute live 500 incident on `/visual-search`.
- Phase 12: the H&M sandbox secret + GCS upload was done via ad hoc, uncaptured `gcloud`
  commands — not even the existing manual process was fully written down.

`scripts/deploy_staging.ps1` already automates steps 2-4 for exactly 3 hardcoded brands
(snitch/fashor/powerlook) as a one-time bootstrap. It is not brand-count-agnostic, has no
GCS-blob-level existence check (confirmed: no code anywhere in the repo checks whether a
remote GCS object already exists before the startup sync tries to download it — 404s are
fatal today), and does not touch `deploy.yml`'s hardcoded `BRANDS_ENABLED` list or
`--set-secrets` entries.

## Decision

Build `scripts/onboard_brand.ps1`, a single re-runnable command that takes over from where
catalog/interaction ingestion (`ingest_catalog.py`, `ingest_interactions.py`,
`build_visual_index_fashionclip.py`, `extract_attributes.py`, `compute_item_colors.py` — all
already run, brand yaml already hand-finished) leaves off, through to an open DRAFT PR ready
for human merge and an optional post-merge deploy trigger.

**Scope boundary respected:** the script never merges a PR and never bypasses
`workflow_dispatch` — merges stay human-only (standing project rule) and the actual
build/push/deploy stays inside the existing, reviewed `deploy.yml`. This is an operational
unblock for us, not a new deploy mechanism.

### Pipeline

1. **Load + validate** `brands/<brand>.yaml` via the real `BrandConfig` pydantic model (not
   raw YAML) — reusing `app/storage.py`'s path-derivation logic (refactored into a shared
   `brand_asset_paths(cfg)` function so the CLI pre-flight and the runtime GCS sync can never
   diverge — the exact failure class that caused the Phase 7 5th boot blocker).
2. **PRE-FLIGHT (always runs first, read-only):**
   - Every required local file exists (catalog parquet, FAISS index dir, embeddings,
     visual index dir, color index, attributes, transaction splits — whichever the yaml
     references). Missing → print the full list, exit 1, touch nothing.
   - Every required path's presence in the target GCS bucket, via
     `gcloud storage objects describe` (read-only) — builds an explicit upload plan
     (NEW vs. already-present) instead of upload-and-hope.
3. **DRY-RUN mode** (`-DryRun`): print the full plan — GCS upload list, secret name +
   whether it already exists, the exact unified diff that would be applied to `deploy.yml`,
   and the exact `gh workflow run` command that would eventually redeploy. Zero mutations,
   exit 0.
4. **Secret Manager**: create `fashion-rec-<brand>-key` if absent (skip if present, `-RotateKey`
   to force), value is a fresh cryptographically-random key (not a predictable placeholder —
   an improvement over `deploy_staging.ps1`'s literal `"<brand>-staging-key"` strings), written
   via the established temp-file-no-BOM-no-newline pattern (the CRLF secret incident lesson).
   Grants `secretmanager.secretAccessor` to the deploy SA (idempotent binding).
5. **GCS upload**: `gcloud storage cp` for every path in the plan.
6. **CONTAINER-VERIFY (mandatory unless `-SkipContainerVerify`, discouraged):** build
   `infra/Dockerfile.cpu` from the current tree, run it with all currently-enabled brands'
   data plus the new brand's data bind-mounted, `BRANDS_ENABLED` including the new brand, and
   confirm over real HTTP inside that container: `/health` lists the brand, and its
   `/similar` (always) and `/recommend` (if `transactions_dir` is set) return 200. Any
   failure stops the script before `deploy.yml` is touched.
7. **`deploy.yml` patch**: append the brand to the `BRANDS_ENABLED` value and add its
   `--set-secrets` entry (both idempotent — skipped if already present), on a fresh branch,
   opened as a **DRAFT PR** (`gh pr create --draft`). The script stops here by default and
   prints the PR URL plus the exact next manual step.
8. **`-TriggerDeploy`** (separate, explicit flag): only runs `gh workflow run` if the PR for
   this brand is confirmed merged; otherwise refuses with a clear message. This is how a
   human "hands back" control to the script after doing the one part that must stay
   human — the merge.
9. **`-VerifyLive`**: after a real deploy has happened (human-triggered or via step 8), curl
   the live revision for the new brand's endpoints and report pass/fail — the same
   merged==live, HTTP-proven standard every prior phase in this project has held to.

### Idempotency

Every mutating step re-checks state before acting: secret existence, GCS object existence
(informational only — upload always overwrites, matching `deploy_staging.ps1`'s existing
behavior, since output is deterministic pipeline data), whether `BRANDS_ENABLED`/`--set-secrets`
already contain the brand, and whether an onboarding PR is already open for this brand slug
(`gh pr list --head onboard-<brand>` — reuses it instead of opening a duplicate). A script
killed mid-run and re-invoked identically cannot double-create a secret, duplicate a PR, or
corrupt `deploy.yml`.

## Consequences

- Closes the GCS-blob-existence gap that no code in the repo currently covers.
- Closes the Phase 12 "un-scripted ad hoc gcloud commands" gap — the H&M sandbox onboarding
  steps are now exactly what this script would have done, captured in source control.
- `deploy_staging.ps1` remains as the one-time bootstrap for the *shared infrastructure*
  (bucket, AR repo, SA, WIF) — `onboard_brand.ps1` assumes that infrastructure already exists
  and only ever adds a brand to it. Not merged into one script: the two have different
  blast radii (one-time infra bootstrap vs. repeatable per-brand operation) and conflating
  them would make the common case (onboard a brand) re-run rarely-needed infra-creation
  logic on every invocation.
- Tier 4 (dynamic registry, server-side ingestion, programmatic client-facing secrets)
  remains explicitly out of scope — this script is still triggered and run by us, not the
  client.

## Alternatives considered

- **Fold into `deploy_staging.ps1`**: rejected — that script is a one-time, 3-brand-hardcoded
  infra bootstrap; genericizing it over `brands/*.yaml` while also adding pre-flight/dry-run/
  container-verify/PR-automation would make one script do two very different jobs with two
  different risk profiles.
- **Python instead of PowerShell**: rejected for consistency — every other operational script
  in this repo that shells out to `gcloud`/`gh` (`deploy_staging.ps1`) is PowerShell, and the
  project is Windows-first. The one Python piece (`brand_asset_paths` path resolution) is
  reused via a thin CLI wrapper so pydantic validation logic isn't duplicated in PowerShell.
- **Auto-merge the PR**: rejected outright — violates the standing "DRAFT PRs, human merge"
  rule with no exception carved out for operational tooling.
