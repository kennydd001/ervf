$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\s100_mtp_inventory.py'
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Runner)) { throw "MTP inventory runner not found: $Runner" }
Write-Host "Repository : $Repo"
Write-Host 'Mode       : CPU/header-only; no CUDA context or tensor payload reads'
$old = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python $Runner
    $code = $LASTEXITCODE
}
finally { $ErrorActionPreference = $old }
if ($code -ne 0) { throw "S100 MTP inventory failed with native exit code $code" }
