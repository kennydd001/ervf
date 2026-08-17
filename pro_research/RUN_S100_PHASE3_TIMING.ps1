param(
    [ValidateSet('qfast','k5','k4','fast_k5','fast_k4','all')]
    [string]$Profile = 'all',
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing runtime environment: $Py" }
$Profiles = if ($Profile -eq 'all') { @('qfast','k5','k4','fast_k5','fast_k4') } else { @($Profile) }
foreach ($P in $Profiles) {
    Write-Host ''; Write-Host "=== S100 phase-3 timing [$P / $Mode] ===" -ForegroundColor Cyan
    $env:PYTHONHASHSEED = '0'
    & $Py 'pro_research\s100_phase3_timing.py' --profile $P --mode $Mode
    if ($LASTEXITCODE -ne 0) { throw "Phase-3 timing profile $P returned $LASTEXITCODE" }
    & $Py 'pro_research\verify_s100_phase3_timing.py' --profile $P
    if ($LASTEXITCODE -ne 0) { throw "Phase-3 timing verifier for $P failed" }
}
