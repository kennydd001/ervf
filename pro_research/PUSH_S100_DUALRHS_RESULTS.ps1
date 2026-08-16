$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-dualrhs') { throw "Expected pro-s100-dualrhs, got '$Branch'" }
$Path = 'pro_research/results/s100_dualrhs'
if (-not (Test-Path $Path)) { throw "No DualRHS result directory: $Path" }
git add -f -- $Path
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --cached --check failed' }
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No DualRHS result changes to push' }
git commit -m 'S100 DualRHS ERVF results'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-s100-dualrhs
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'DualRHS results pushed.' -ForegroundColor Green
