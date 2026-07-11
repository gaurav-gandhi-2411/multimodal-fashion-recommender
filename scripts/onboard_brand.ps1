<#
  onboard_brand.ps1 -- repeatable, per-brand onboarding for the fashion-recommender STAGING
  deploy in the SHARED project iconic-reactor-496423-m4.

  Unlike scripts/deploy_staging.ps1 (a one-time, 3-brand-hardcoded bootstrap of the SHARED
  infrastructure: bucket, AR repo, SA, WIF), this script assumes that infrastructure already
  exists and does ONE thing, repeatably, for any brand whose brands/<brand>.yaml has already
  been hand-finished by scripts/ingest_catalog.py (+ ingest_interactions.py, build_visual_
  index_fashionclip.py, extract_attributes.py, compute_item_colors.py as applicable):

    pre-flight -> (dry-run report, if asked) -> Secret Manager key -> GCS upload ->
    container-verify (mandatory) -> deploy.yml patch as a DRAFT PR -> stop.

  Merges stay human-only (standing project rule). -TriggerDeploy and -VerifyLive are
  separate, explicit, later invocations of this SAME script after a human has merged the PR.

  Full rationale: docs/architecture/adr/0001-brand-onboarding-runbook.md

  Safety model:
    * Every mutating step re-checks state first (secret existence, GCS-blob existence,
      whether deploy.yml already contains the brand, whether an onboarding PR is already
      open) -- a killed-and-re-run script cannot double-create a secret, duplicate a PR, or
      corrupt deploy.yml.
    * -DryRun is a HARD guarantee of zero mutation: it exits before Secret Manager, GCS
      upload, docker, or git are touched.
    * Container-verify (step 5) is mandatory unless -SkipContainerVerify is passed --
      matches this project's standing Deploy Verification Standard (PROJECT_MEMORY.md):
      "local pass != container pass."
    * Never touches aetherart-497918: $Project defaults to iconic-reactor-496423-m4 only;
      nothing in this script can default to any other project.

  Run from the repo ROOT. Requires: gcloud + gh already authenticated, docker running.
#>
param(
  [Parameter(Mandatory = $true)][string]$Brand,
  [switch]$DryRun,
  [switch]$RotateKey,
  [switch]$SkipContainerVerify,
  [switch]$TriggerDeploy,
  [switch]$VerifyLive,
  [string]$Project = "iconic-reactor-496423-m4",
  [string]$Region = "asia-south1",
  [string]$Bucket = "fashion-rec-staging-iconic-reactor-496423-m4",
  [string]$Repo = "gaurav-gandhi-2411/multimodal-fashion-recommender"
)

$ErrorActionPreference = "Stop"
trap { Write-Host "`n*** HALTED: $_" -ForegroundColor Red; exit 1 }

# ----------------------------------------------------------------------------- config
$Brand = $Brand.ToLower()
$SA_EMAIL = "fashion-rec-deployer@$Project.iam.gserviceaccount.com"
$CLOUD_RUN_SERVICE = "fashion-recommender-staging"
$secretName = "fashion-rec-$Brand-key"
$onboardBranch = "onboard-$Brand"
$deployYmlPath = ".github/workflows/deploy.yml"

# ----------------------------------------------------------------------------- helpers
function Exists([string[]]$DescribeArgs) { & gcloud @DescribeArgs *> $null; return ($LASTEXITCODE -eq 0) }
function Step([string]$Desc, [scriptblock]$Action) {
  Write-Host "==> $Desc" -ForegroundColor Cyan
  & $Action
  if ($LASTEXITCODE -ne 0) { throw "failed: $Desc (exit $LASTEXITCODE)" }
}
function Skip([string]$What) { Write-Host "  skip (exists): $What" -ForegroundColor DarkGray }

# Regex-parse the handful of top-level scalar fields we need out of a brand YAML without
# taking a new YAML-parsing dependency in PowerShell -- every brand yaml in this repo keeps
# these fields flat and unindented at the top level (confirmed against brands/*.yaml).
function Get-BrandYamlFields([string]$Path) {
  $content = Get-Content $Path -Raw
  $slug = if ($content -match '(?m)^brand:\s*(\S+)') { $Matches[1] } else { $null }
  $apiKeyEnv = if ($content -match '(?m)^api_key_env:\s*(\S+)') { $Matches[1] } else { $null }
  $catalogPath = if ($content -match '(?m)^catalog_path:\s*(\S+)') { $Matches[1] } else { $null }
  $hasTransactions = $content -match '(?m)^transactions_dir:\s*\S+'
  [PSCustomObject]@{
    Slug            = $slug
    ApiKeyEnv       = $apiKeyEnv
    CatalogPath     = $catalogPath
    HasTransactions = [bool]$hasTransactions
  }
}

# Reads the first article_id out of a brand's catalog parquet -- used to exercise
# /similar (and /recommend via item_id) with a real ID instead of a guessed one.
function Get-SampleArticleId([string]$CatalogPath) {
  $out = python -c "import pandas as pd; df = pd.read_parquet(r'$CatalogPath'); print(int(df['article_id'].iloc[0]))"
  if ($LASTEXITCODE -ne 0 -or -not $out) { throw "Could not read a sample article_id from $CatalogPath" }
  return $out.Trim()
}

function Test-PortFree([int]$Port) {
  try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
    $listener.Stop()
    return $true
  } catch { return $false }
}
function Get-FreeLocalPort([int]$Start = 18080) {
  $port = $Start
  while (-not (Test-PortFree $port)) { $port++ }
  return $port
}

# Invoke-WebRequest throws a terminating error on non-2xx by default (in both Windows
# PowerShell 5.1 and PS7) -- this helper always returns a status code instead of throwing,
# so callers can make an explicit pass/fail decision.
function Invoke-HttpStatus {
  param([string]$Uri, [string]$Method = "GET", [hashtable]$Headers = @{}, [string]$Body = $null)
  try {
    $params = @{ Uri = $Uri; Method = $Method; Headers = $Headers; UseBasicParsing = $true; TimeoutSec = 15 }
    if ($Body) { $params.Body = $Body }
    $resp = Invoke-WebRequest @params
    return [int]$resp.StatusCode
  } catch {
    if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode.value__ }
    return -1  # connection-level failure (container not up yet, etc.)
  }
}

# Computes the deploy.yml patch in-memory: append $Brand to BRANDS_ENABLED and its
# --set-secrets entry. Idempotent by design (AlreadyPresent=true short-circuits both).
# Shared by -DryRun's diff preview and step 6's real patch so they can never disagree.
function Get-DeployYmlPatch {
  param([string]$Content, [string]$Brand)

  if ($Content -notmatch 'BRANDS_ENABLED=([a-z0-9_,]+)"') {
    throw "deploy.yml: could not locate BRANDS_ENABLED= in --set-env-vars line."
  }
  $currentBrandsValue = $Matches[1]
  $brandAlreadyEnabled = ($currentBrandsValue -split ',') -contains $Brand

  if ($Content -notmatch '--set-secrets "([^"]+)"') {
    throw "deploy.yml: could not locate --set-secrets line."
  }
  $currentSecretsValue = $Matches[1]
  $secretEnvVar = "$($Brand.ToUpper())_API_KEY"
  $secretAlreadyPresent = [bool](($currentSecretsValue -split ',') | Where-Object { $_.StartsWith("$secretEnvVar=") })

  $alreadyPresent = $brandAlreadyEnabled -and $secretAlreadyPresent
  $patched = $Content
  if (-not $brandAlreadyEnabled) {
    $newBrandsValue = "$currentBrandsValue,$Brand"
    $patched = $patched.Replace("BRANDS_ENABLED=$currentBrandsValue`"", "BRANDS_ENABLED=$newBrandsValue`"")
  }
  if (-not $secretAlreadyPresent) {
    $newSecretsValue = "$currentSecretsValue,$secretEnvVar=fashion-rec-$Brand-key:latest"
    $patched = $patched.Replace("--set-secrets `"$currentSecretsValue`"", "--set-secrets `"$newSecretsValue`"")
  }

  [PSCustomObject]@{
    AlreadyPresent  = $alreadyPresent
    Patched         = $patched
    BrandEnabled    = $brandAlreadyEnabled
    SecretPresent   = $secretAlreadyPresent
  }
}

# ----------------------------------------------------------------------------- 0. sanity
Write-Host "`n=== 0. Environment sanity ===" -ForegroundColor Yellow
if (-not (Test-Path "brands")) { throw "Run this from the repo ROOT (brands/ not found here)." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud not on PATH." }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "gh (GitHub CLI) not on PATH." }
$acct = (gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null)
if (-not $acct) { throw "gcloud is not authenticated. Run:  gcloud auth login" }
gh auth status *> $null; if ($LASTEXITCODE -ne 0) { throw "gh is not authenticated. Run:  gh auth login" }
Write-Host "  gcloud account: $acct"
Step "set active project to $Project" { gcloud config set project $Project }
$originalBranch = (git rev-parse --abbrev-ref HEAD).Trim()

# ----------------------------------------------------------------------------- 1. preflight (always, read-only)
Write-Host "`n=== 1. Pre-flight: local assets (read-only) ===" -ForegroundColor Yellow
$preflightStderr = New-TemporaryFile
$preflightStdout = & python scripts/brand_preflight.py --brand $Brand --brands-dir brands --output-base . 2>$preflightStderr.FullName
Get-Content $preflightStderr.FullName -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
Remove-Item $preflightStderr.FullName -ErrorAction SilentlyContinue

if (-not $preflightStdout) {
  Write-Host "HALT: brand_preflight.py produced no output -- brands/$Brand.yaml likely missing or invalid. See diagnostics above." -ForegroundColor Red
  exit 1
}
$preflight = $preflightStdout | ConvertFrom-Json
if (-not $preflight.all_present) {
  Write-Host "HALT: missing local assets for brand '$Brand' -- fix before onboarding. Touching nothing." -ForegroundColor Red
  $preflight.missing_local | ForEach-Object { Write-Host "    - $_" }
  exit 1
}
Write-Host "  All $($preflight.required_paths.Count) required local assets present for '$Brand'." -ForegroundColor Green

Write-Host "`n=== 1b. Pre-flight: GCS existence (read-only) ===" -ForegroundColor Yellow
$toUploadNew = @()
$toOverwrite = @()
foreach ($path in $preflight.required_paths) {
  if (Exists @("storage", "objects", "describe", "gs://$Bucket/$path", "--project", $Project)) {
    $toOverwrite += $path
  } else {
    $toUploadNew += $path
  }
}
Write-Host "  Pre-flight report for brand '$Brand':"
Write-Host "    Required paths          : $($preflight.required_paths.Count)"
Write-Host "    Local missing           : 0"
Write-Host "    GCS new (to upload)     : $($toUploadNew.Count)"
$toUploadNew | ForEach-Object { Write-Host "      + $_" -ForegroundColor DarkGray }
Write-Host "    GCS existing (overwrite): $($toOverwrite.Count)"
$toOverwrite | ForEach-Object { Write-Host "      = $_" -ForegroundColor DarkGray }

# Brand-field metadata needed later (container-verify + verify-live); hoisted here so it's
# available even when -SkipContainerVerify is passed.
$allBrandFields = Get-ChildItem "brands/*.yaml" | ForEach-Object { Get-BrandYamlFields $_.FullName }
$newBrandFields = $allBrandFields | Where-Object { $_.Slug -eq $Brand } | Select-Object -First 1
if (-not $newBrandFields) { throw "brands/$Brand.yaml has no 'brand:' field matching '$Brand' (should have been caught by pre-flight)." }
$brandsEnabledValue = ($allBrandFields.Slug | Select-Object -Unique) -join ','

# ----------------------------------------------------------------------------- 2. dry run
if ($DryRun) {
  Write-Host "`n=== DRY RUN -- read-only, zero mutations will be made ===" -ForegroundColor Yellow

  $secretExists = Exists @("secrets", "describe", $secretName, "--project", $Project)
  $secretAction = if (-not $secretExists) { "would CREATE" }
                  elseif ($RotateKey) { "would ROTATE (add new version)" }
                  else { "exists -- no-op (pass -RotateKey to force rotation)" }
  Write-Host "  Secret Manager: $secretName -> $secretAction"

  Write-Host "  Deploy trigger command (run only after the PR below is merged):"
  Write-Host "    gh workflow run `"Deploy to Cloud Run`" --repo $Repo --ref main -f environment=staging"

  $originalContent = [IO.File]::ReadAllText((Resolve-Path $deployYmlPath))
  $patch = Get-DeployYmlPatch -Content $originalContent -Brand $Brand
  if ($patch.AlreadyPresent) {
    Write-Host "  deploy.yml: '$Brand' already present in BRANDS_ENABLED + --set-secrets -- no patch would be applied."
  } else {
    $tmpPatched = New-TemporaryFile
    [IO.File]::WriteAllText($tmpPatched.FullName, $patch.Patched)
    Write-Host "  deploy.yml diff that WOULD be applied (tracked file untouched):"
    git diff --no-index -- $deployYmlPath $tmpPatched.FullName
    Remove-Item $tmpPatched.FullName -ErrorAction SilentlyContinue
  }

  Write-Host "`nDry run complete -- zero mutations made." -ForegroundColor Green
  exit 0
}

# ----------------------------------------------------------------------------- 3. secret manager
Write-Host "`n=== 3. Secret Manager ===" -ForegroundColor Yellow
$secretExists = Exists @("secrets", "describe", $secretName, "--project", $Project)
$printedKey = $null
if (-not $secretExists) {
  $printedKey = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(24))
  $tmp = New-TemporaryFile
  [IO.File]::WriteAllText($tmp.FullName, $printedKey)  # exact bytes, no trailing newline/BOM
  Step "create secret $secretName" { gcloud secrets create $secretName --data-file=$($tmp.FullName) --project $Project }
  Remove-Item $tmp.FullName
} elseif ($RotateKey) {
  $printedKey = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(24))
  $tmp = New-TemporaryFile
  [IO.File]::WriteAllText($tmp.FullName, $printedKey)
  Step "rotate secret $secretName (new version)" { gcloud secrets versions add $secretName --data-file=$($tmp.FullName) --project $Project }
  Remove-Item $tmp.FullName
} else {
  Skip "secret $secretName (pass -RotateKey to force rotation)"
}
Step "grant secretAccessor on $secretName to $SA_EMAIL" {
  gcloud secrets add-iam-policy-binding $secretName --project $Project --member="serviceAccount:$SA_EMAIL" --role=roles/secretmanager.secretAccessor
}
if ($printedKey) {
  Write-Host "`n  Generated key for '$Brand' -- SAVE THIS, it will not be shown again:" -ForegroundColor Yellow
  Write-Host "    $printedKey" -ForegroundColor White
}
# Needed for container-verify below regardless of whether we just created/rotated it.
$brandApiKeyValue = (gcloud secrets versions access latest --secret=$secretName --project $Project)

# ----------------------------------------------------------------------------- 4. GCS upload
Write-Host "`n=== 4. GCS upload ===" -ForegroundColor Yellow
foreach ($path in $preflight.required_paths) {
  Step "upload $path" { gcloud storage cp $path "gs://$Bucket/$path" }
}

# ----------------------------------------------------------------------------- 5. container-verify
if (-not $SkipContainerVerify) {
  Write-Host "`n=== 5. Container-verify (mandatory) ===" -ForegroundColor Yellow
  $testImage = "fashion-rec-onboard-test"
  $testContainer = "fashion-rec-onboard-test-$Brand"
  $testPort = Get-FreeLocalPort

  Step "docker build $testImage" { docker build -f infra/Dockerfile.cpu -t $testImage . }

  $dataPath = (Resolve-Path "data").Path
  $indicesPath = (Resolve-Path "indices").Path
  $checkpointsPath = (Resolve-Path "checkpoints").Path
  $brandsPath = (Resolve-Path "brands").Path

  # Every currently-enabled brand needs *some* API key env var set or registry._load_brand
  # raises at startup; only the new brand needs the REAL key (to exercise /similar for real).
  $envArgs = @()
  foreach ($f in $allBrandFields) {
    $keyValue = if ($f.Slug -eq $Brand) { $brandApiKeyValue } else { "dev-$($f.Slug)-key" }
    $envArgs += @("-e", "$($f.ApiKeyEnv)=$keyValue")
  }
  $dockerRunArgs = @(
    "run", "-d", "--name", $testContainer,
    "-p", "${testPort}:8000",
    "-v", "${dataPath}:/app/data:ro",
    "-v", "${indicesPath}:/app/indices:ro",
    "-v", "${checkpointsPath}:/app/checkpoints:ro",
    "-v", "${brandsPath}:/app/brands:ro",
    "-e", "BRANDS_ENABLED=$brandsEnabledValue"
  ) + $envArgs + @($testImage)

  try {
    Step "docker run test container ($testContainer on port $testPort)" { docker @dockerRunArgs }

    # Poll /health for readiness. NOTE (deviation flagged in the hand-back report):
    # HealthResponse today only returns {"status": "ok"} -- no brand list (HealthBrand
    # exists in app/api/schemas.py but is unused/dead code). load_registry() loads every
    # enabled brand eagerly during lifespan startup, so a broken new-brand config would
    # crash the WHOLE app before /health ever responds -- making "/health responds within
    # the timeout" a valid multi-brand-boot signal. The brand-SPECIFIC proof is the
    # mandatory /similar (and /recommend) call directly below.
    $healthy = $false
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
      if ((Invoke-HttpStatus -Uri "http://localhost:$testPort/health") -eq 200) { $healthy = $true; break }
      Start-Sleep -Seconds 3
    }
    if (-not $healthy) { throw "Container did not become healthy within 60s. Check: docker logs $testContainer" }
    Write-Host "  /health -> 200 (container up, all enabled brands including '$Brand' loaded)" -ForegroundColor Green

    $articleId = Get-SampleArticleId -CatalogPath $newBrandFields.CatalogPath
    $similarStatus = Invoke-HttpStatus -Uri "http://localhost:$testPort/v1/$Brand/item/$articleId/similar" -Headers @{ "X-Api-Key" = $brandApiKeyValue }
    if ($similarStatus -ne 200) { throw "/v1/$Brand/item/$articleId/similar returned $similarStatus inside container-verify." }
    Write-Host "  /v1/$Brand/item/$articleId/similar -> 200 (inside container)" -ForegroundColor Green

    if ($newBrandFields.HasTransactions) {
      $recBody = @{ item_id = "$articleId" } | ConvertTo-Json -Compress
      $recStatus = Invoke-HttpStatus -Uri "http://localhost:$testPort/v1/$Brand/recommend" -Method "POST" `
        -Headers @{ "X-Api-Key" = $brandApiKeyValue; "Content-Type" = "application/json" } -Body $recBody
      if ($recStatus -ne 200) { throw "/v1/$Brand/recommend returned $recStatus inside container-verify." }
      Write-Host "  /v1/$Brand/recommend -> 200 (inside container)" -ForegroundColor Green
    }

    $containerVerifyResult = "PASSED"
  } finally {
    docker stop $testContainer *> $null
    docker rm $testContainer *> $null
  }
} else {
  Write-Host "`n=== 5. Container-verify SKIPPED (-SkipContainerVerify) -- discouraged ===" -ForegroundColor DarkYellow
  $containerVerifyResult = "SKIPPED (-SkipContainerVerify passed)"
}

# ----------------------------------------------------------------------------- 6. deploy.yml patch + draft PR
Write-Host "`n=== 6. deploy.yml patch + draft PR ===" -ForegroundColor Yellow
$originalContent = [IO.File]::ReadAllText((Resolve-Path $deployYmlPath))
$patch = Get-DeployYmlPatch -Content $originalContent -Brand $Brand

if ($patch.AlreadyPresent) {
  Write-Host "  '$Brand' already present in deploy.yml (BRANDS_ENABLED + --set-secrets) -- nothing to patch." -ForegroundColor DarkGray
} else {
  $existingPr = gh pr list --head $onboardBranch --repo $Repo --json "url,state" | ConvertFrom-Json
  if ($existingPr -and $existingPr.Count -gt 0) {
    Write-Host "  Onboarding PR already open for '$Brand': $($existingPr[0].url)" -ForegroundColor DarkGray
  } else {
    # infra/Dockerfile.cpu does `COPY brands/ brands/` at build time -- data/indices/checkpoints
    # are gitignored and only ever reach the container via the runtime GCS sync, but
    # brands/<brand>.yaml is baked into the image. Without committing it alongside deploy.yml,
    # BRANDS_ENABLED would list the brand while the image silently never contains its config --
    # the exact silent-no-op failure class this project has been burned by before.
    $brandYamlPath = "brands/$Brand.yaml"
    if (-not (Test-Path $brandYamlPath)) {
      throw "brands/$Brand.yaml not found -- run ingest_catalog.py (and friends) first; this script onboards an ALREADY-ingested brand."
    }

    # Working-tree safety: refuse to mix unrelated dirty state into the onboarding commit.
    # Scoped to exactly the two paths this step commits -- deploy.yml and the brand's own
    # yaml -- so pre-existing unrelated untracked/modified files elsewhere in the repo don't
    # falsely block onboarding (this script must not silently sweep up someone else's WIP).
    $dirty = git status --porcelain -- $deployYmlPath $brandYamlPath
    $unexpectedDirty = $dirty | Where-Object { $_ -notmatch [regex]::Escape($brandYamlPath) -or $_ -notmatch '^\?\? ' }
    if ($unexpectedDirty) {
      throw "$deployYmlPath has pre-existing local modifications -- commit or stash before onboarding.`n$unexpectedDirty"
    }

    $switchedBranch = $false
    try {
      $localBranchExists = [bool](git branch --list $onboardBranch)
      if ($localBranchExists) {
        Step "checkout existing local branch $onboardBranch (partial-run resume)" { git checkout $onboardBranch }
      } else {
        Step "create branch $onboardBranch" { git checkout -b $onboardBranch }
      }
      $switchedBranch = $true

      [IO.File]::WriteAllText((Resolve-Path $deployYmlPath), $patch.Patched)
      $pendingDiff = git status --porcelain -- $deployYmlPath $brandYamlPath
      if ($pendingDiff) {
        Step "stage deploy.yml + brands/$Brand.yaml" { git add $deployYmlPath $brandYamlPath }
        Step "commit deploy.yml + brands/$Brand.yaml" { git commit -m "feat(deploy): onboard $Brand brand" }
      } else {
        Write-Host "  deploy.yml + brands/$Brand.yaml already committed on $onboardBranch (partial-run resume)." -ForegroundColor DarkGray
      }
      Step "push $onboardBranch" { git push -u origin $onboardBranch }

      $prBody = @"
Onboards brand '$Brand' via scripts/onboard_brand.ps1.

Pre-flight: $($preflight.required_paths.Count) required assets -- $($toUploadNew.Count) newly uploaded to GCS, $($toOverwrite.Count) already present.
Container-verify: $containerVerifyResult
"@
      $prUrl = gh pr create --draft --repo $Repo --title "feat(deploy): onboard $Brand brand" --body $prBody --head $onboardBranch --base main
      Write-Host "  Draft PR opened: $prUrl" -ForegroundColor Green
    } finally {
      if ($switchedBranch) { Step "return to $originalBranch" { git checkout $originalBranch } }
    }
  }
  Write-Host "`nNext: review + merge the PR, then re-run this script with -TriggerDeploy to trigger the Cloud Run redeploy." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------- 7. trigger deploy
if ($TriggerDeploy) {
  Write-Host "`n=== 7. Trigger deploy (-TriggerDeploy) ===" -ForegroundColor Yellow
  $prState = (gh pr view $onboardBranch --repo $Repo --json state | ConvertFrom-Json).state
  if ($prState -ne "MERGED") {
    Write-Host "REFUSING: PR for '$onboardBranch' is not yet merged (state: $prState). Merge it first, then re-run with -TriggerDeploy." -ForegroundColor Red
    exit 1
  }
  Step "trigger 'Deploy to Cloud Run' workflow" { gh workflow run "Deploy to Cloud Run" --repo $Repo --ref main -f environment=staging }
  Write-Host "  Triggered. Watch with:  gh run watch --repo $Repo" -ForegroundColor Green
}

# ----------------------------------------------------------------------------- 8. verify live
if ($VerifyLive) {
  Write-Host "`n=== 8. Verify live (-VerifyLive) ===" -ForegroundColor Yellow
  $serviceUrl = (gcloud run services describe $CLOUD_RUN_SERVICE --region $Region --project $Project --format "value(status.url)")
  if (-not $serviceUrl) { Write-Host "FAIL: could not resolve Cloud Run service URL." -ForegroundColor Red; exit 1 }

  $articleId = Get-SampleArticleId -CatalogPath $newBrandFields.CatalogPath
  $liveKey = (gcloud secrets versions access latest --secret=$secretName --project $Project)
  $liveUrl = "$serviceUrl/v1/$Brand/item/$articleId/similar"
  $status = Invoke-HttpStatus -Uri $liveUrl -Headers @{ "X-Api-Key" = $liveKey }

  if ($status -eq 200) {
    Write-Host "  PASS: $liveUrl -> 200" -ForegroundColor Green
  } else {
    Write-Host "  FAIL: $liveUrl -> $status" -ForegroundColor Red
    exit 1
  }
}
