$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv-nemotron\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Missing Nemotron venv: $Py"
}

Write-Host "=== S100 K2 V18 target oracle ==="
Write-Host "Branch should be: pro-s100-k2-oracle-v18"
& git status --short
& git rev-parse --abbrev-ref HEAD
& git rev-parse HEAD

$Mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
if ($Mode -ne "smoke" -and $Mode -ne "full") {
    throw "Usage: .\pro_research\RUN_S100_K2_V18_TARGET_ORACLE.ps1 [smoke|full]"
}

Write-Host "Running mode=$Mode"
& $Py "pro_research\s100_k2_v18_target_oracle.py" --mode $Mode
if ($LASTEXITCODE -ne 0) {
    Write-Host "Runner returned $LASTEXITCODE. Result artifact is preserved; verifier will still be attempted if JSON exists."
}

$Result = Join-Path $Repo "pro_research\results\s100_k2_v18\PRO_S100_K2_V18_TARGET_ORACLE.json"
if (Test-Path $Result) {
    & $Py "pro_research\verify_s100_k2_v18_target_oracle.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Independent verifier failed with exit code $LASTEXITCODE"
    }
} else {
    throw "Runner produced no result JSON: $Result"
}

Write-Host "=== Result ==="
Get-Content $Result | Select-String '"status"|"k2_p50_ms_per_2tok"|"k2_effective_verified_tok_s"|"k2_speedup_vs_seq_mid"|"P1_K2_block_lt_19_285ms"|"P2_K2_block_lt_17_500ms"'
