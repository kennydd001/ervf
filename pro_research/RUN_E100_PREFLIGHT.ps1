$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Check = Join-Path $Repo 'pro_research\preflight_e100_cpu.py'
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Check)) { throw "Preflight not found: $Check" }
Write-Host "Repository : $Repo"
Write-Host 'Mode       : CPU-only; no CUDA context should be created'
$old = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python $Check
    $code = $LASTEXITCODE
}
finally { $ErrorActionPreference = $old }
if ($code -ne 0) { throw "E100 CPU preflight failed with native exit code $code" }
Write-Host 'E100 CPU preflight passed.' -ForegroundColor Green
