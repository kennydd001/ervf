$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-dualrhs') { throw "Expected pro-s100-dualrhs, got '$Branch'" }
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing venv: $Py" }
$Mode = if ($args.Count -gt 0) { $args[0] } else { 'smoke' }
if ($Mode -notin @('smoke','full')) { throw '.\pro_research\RUN_S100_DUALRHS_ERVF.ps1 [smoke|full]' }

$Dir = Join-Path $Repo 'pro_research\results\s100_dualrhs'
$Result = Join-Path $Dir 'PRO_S100_DUALRHS_ERVF.json'
$Verify = Join-Path $Dir 'PRO_S100_DUALRHS_ERVF_VERIFICATION.json'
$SmokeResult = Join-Path $Dir 'PRO_S100_DUALRHS_ERVF_smoke.json'
$SmokeVerify = Join-Path $Dir 'PRO_S100_DUALRHS_ERVF_VERIFICATION_smoke.json'

Write-Host '=== S100 DualRHS-ERVF ==='
git status --short
git rev-parse HEAD
$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "Preflight GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

# FULL is authorized only by a preserved, independently verified smoke pass.
if ($Mode -eq 'full') {
    if (-not (Test-Path $SmokeResult) -or -not (Test-Path $SmokeVerify)) {
        throw 'No preserved verified smoke result. Run smoke first; full is fail-closed.'
    }
    $sr = Get-Content $SmokeResult -Raw | ConvertFrom-Json
    $sv = Get-Content $SmokeVerify -Raw | ConvertFrom-Json
    if ($sr.mode -ne 'smoke' -or $sr.status -ne 'smoke_pass' -or -not $sv.passed) {
        throw "Smoke did not authorize full: mode=$($sr.mode) status=$($sr.status) verifier_passed=$($sv.passed)"
    }
}

# Freeze Python's name hashes before process start so every synthetic activation
# seed in this preregistered benchmark is reproducible across runs/machines.
$env:PYTHONHASHSEED = '0'

& $Py 'pro_research\s100_dualrhs_entry.py' --mode $Mode
$rc = $LASTEXITCODE
if (-not (Test-Path $Result)) { throw "No result JSON produced (runner rc=$rc)" }
if ($rc -ne 0) {
    Write-Host "Runner returned $rc; showing preserved result/error." -ForegroundColor Yellow
    Get-Content $Result | Select-String '"status"|"message"|"traceback"'
    exit $rc
}

& $Py 'pro_research\verify_s100_dualrhs_ervf.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent DualRHS verifier failed' }
if (-not (Test-Path $Verify)) { throw 'Verifier produced no JSON' }

if ($Mode -eq 'smoke') {
    $r = Get-Content $Result -Raw | ConvertFrom-Json
    $v = Get-Content $Verify -Raw | ConvertFrom-Json
    if ($r.status -eq 'smoke_pass' -and $v.passed) {
        Copy-Item $Result $SmokeResult -Force
        Copy-Item $Verify $SmokeVerify -Force
        Write-Host 'Smoke PASS preserved; full is now authorized.' -ForegroundColor Green
    } else {
        Write-Host "Smoke did not authorize full: status=$($r.status), verifier=$($v.passed)" -ForegroundColor Yellow
    }
}

Write-Host '=== Result ==='
Get-Content $Result | Select-String '"status"|"weighted_speedup"|"projected_common_projection_saving_ms_per_K2_block"|"projection_only_projected_K2_block_ms_from_38_67655"|"integration_open"'
