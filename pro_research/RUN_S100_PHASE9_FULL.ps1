# S100 phase 9 — one-click testpack.
#
# Draait de volledige gerepareerde phase-9-suite in de gepreregistreerde
# volgorde (RTX/Arc miss-probes, capacity A/C/C/B per profiel, compares,
# miss-economics, summary, repair-verify) als verse subprocessen, met
# hergebruik van de bestaande 8192-token trace en UPMISS NPZ's.
#
# Gebruik:
#   powershell -ExecutionPolicy Bypass -File pro_research\RUN_S100_PHASE9_FULL.ps1
#   ... [-SkipCapacity] [-SkipProbes]
#
# Exitcode 0 = elke stap groen of een schoon, compleet `infeasible_vram`-
# verdict; 1 = minstens één technical_failure.

param(
    [switch]$SkipCapacity,
    [switch]$SkipProbes
)

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    throw "Missing runtime Python: $Py"
}

$RunnerArgs = @('pro_research\run_s100_phase9_full.py')
if ($SkipCapacity) { $RunnerArgs += '--skip-capacity' }
if ($SkipProbes) { $RunnerArgs += '--skip-probes' }

& $Py @RunnerArgs
$Code = $LASTEXITCODE

$Summary = 'pro_research\results\s100_phase9\S100_PHASE9_SUMMARY.txt'
if (Test-Path $Summary) {
    Write-Host ''
    Get-Content $Summary | ForEach-Object { Write-Host "  $_" }
}

exit $Code
