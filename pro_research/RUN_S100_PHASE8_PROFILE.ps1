
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
& $Py 'pro_research\s100_phase8_route_profile.py'
if ($LASTEXITCODE -ne 0) {
    throw "Phase-8 route profile failed: $LASTEXITCODE"
}
