$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-dualrhs') { throw "Expected pro-s100-dualrhs, got '$Branch'" }
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing venv: $Py" }
$Mode = if ($args.Count -gt 0) { $args[0] } else { 'smoke' }
if ($Mode -notin @('smoke','full')) { throw 'Usage: .\pro_research\RUN_S100_DUALRHS_ERVF.ps1 [smoke|full]' }

Write-Host '=== S100 DualRHS-ERVF ==='
git status --short
git rev-parse HEAD
$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "Preflight GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

& $Py 'pro_research\s100_dualrhs_ervf.py' --mode $Mode
$rc = $LASTEXITCODE
$Result = Join-Path $Repo 'pro_research\results\s100_dualrhs\PRO_S100_DUALRHS_ERVF.json'
if (-not (Test-Path $Result)) { throw "No result JSON produced (runner rc=$rc)" }
if ($rc -ne 0) {
    Write-Host "Runner returned $rc; showing preserved result/error." -ForegroundColor Yellow
    Get-Content $Result | Select-String '"status"|"message"|"traceback"'
    exit $rc
}

& $Py 'pro_research\verify_s100_dualrhs_ervf.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent DualRHS verifier failed' }
Write-Host '=== Result ==='
Get-Content $Result | Select-String '"status"|"weighted_speedup"|"projected_common_projection_saving_ms_per_K2_block"|"projection_only_projected_K2_block_ms_from_38_67655"|"integration_open"'
