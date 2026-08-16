$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv-nemotron\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Missing Nemotron venv: $Py"
}

Write-Host "=== S100 K2 V18 target oracle ==="
$Branch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($Branch -ne "pro-s100-k2-oracle-v18") {
    throw "Wrong branch '$Branch'; expected pro-s100-k2-oracle-v18"
}
& git status --short
& git rev-parse HEAD

$Mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
if ($Mode -ne "smoke" -and $Mode -ne "full") {
    throw "Usage: .\pro_research\RUN_S100_K2_V18_TARGET_ORACLE.ps1 [smoke|full]"
}

# Stronger than common.py's compute-app check on Windows/WDDM: some CUDA jobs do
# not appear in --query-compute-apps, but their VRAM use is still visible.  The
# Lightning runner should start from an essentially idle dGPU.  This guard is
# intentionally conservative so a concurrent Kimi/Claude test cannot invalidate
# timing or exhaust the V18 plane gate.
try {
    $UsedLine = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    $UsedMiB = [int]$UsedLine
    Write-Host "Preflight GPU memory.used = $UsedMiB MiB"
    if ($UsedMiB -gt 1024) {
        throw "GPU is not idle ($UsedMiB MiB already used). Another test/process may still be running. Stop it and retry."
    }
} catch {
    if ($_.Exception.Message -like "GPU is not idle*") { throw }
    Write-Warning "Could not enforce memory.used preflight: $($_.Exception.Message)"
}

$ResultDir = Join-Path $Repo "pro_research\results\s100_k2_v18"
$Result = Join-Path $ResultDir "PRO_S100_K2_V18_TARGET_ORACLE.json"

# Full is not allowed to overwrite/obscure a failed smoke.  It may proceed only
# after a smoke reached a scientifically interpretable (correct + stable) status.
if ($Mode -eq "full") {
    if (-not (Test-Path $Result)) {
        throw "No prior smoke result exists. Run smoke first."
    }
    $Prev = Get-Content $Result -Raw | ConvertFrom-Json
    $Allowed = @("k2_v18_feasible_candidate", "k2_v18_below_s100_gate", "layer_major_v18_negative")
    if ($Prev.mode -ne "smoke" -or $Allowed -notcontains [string]$Prev.status) {
        throw "Refusing full: previous result is mode='$($Prev.mode)' status='$($Prev.status)'. Fix/re-run smoke first."
    }
}

# Preserve every prior result before the Python runner writes its canonical path.
if (Test-Path $Result) {
    $Hist = Join-Path $ResultDir "history"
    New-Item -ItemType Directory -Force -Path $Hist | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    Copy-Item $Result (Join-Path $Hist "PRO_S100_K2_V18_TARGET_ORACLE_$Stamp.json")
}

Write-Host "Running mode=$Mode"
& $Py "pro_research\s100_k2_v18_target_oracle.py" --mode $Mode
$RunnerRC = $LASTEXITCODE

if (-not (Test-Path $Result)) {
    throw "Runner produced no result JSON: $Result"
}

$Source = Get-Content $Result -Raw | ConvertFrom-Json
if ($RunnerRC -ne 0 -or [string]$Source.status -eq "technical_failure") {
    Write-Host "=== Technical/runner failure preserved ===" -ForegroundColor Yellow
    Get-Content $Result | Select-String '"status"|"error"|"message"|"traceback"'
    throw "K2 runner failed (rc=$RunnerRC, status=$($Source.status)). Do NOT run full; diagnose smoke first."
}

& $Py "pro_research\verify_s100_k2_v18_target_oracle.py"
if ($LASTEXITCODE -ne 0) {
    throw "Independent verifier failed with exit code $LASTEXITCODE"
}

Write-Host "=== Result ==="
Get-Content $Result | Select-String '"status"|"k2_p50_ms_per_2tok"|"k2_effective_verified_tok_s"|"k2_speedup_vs_seq_mid"|"P1_K2_block_lt_19_285ms"|"P2_K2_block_lt_17_500ms"'
