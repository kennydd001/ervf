$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
& $Py 'pro_research\s100_phase6_heldout.py'
if ($LASTEXITCODE -ne 0) { throw "Phase-6 heldout failed: $LASTEXITCODE" }
