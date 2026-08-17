$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing $Py" }
& $Py 'pro_research\s100_phase6_preflight.py'
if ($LASTEXITCODE -ne 0) { throw "Phase-6 preflight failed: $LASTEXITCODE" }
