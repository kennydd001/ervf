$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-k2-oracle-v18') { throw "Expected pro-s100-k2-oracle-v18, got '$Branch'" }
$Path = 'pro_research/results/s100_k2_v18'
if (-not (Test-Path $Path)) { throw "No K2 result directory: $Path" }
git add -f -- $Path
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --cached --check failed' }
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No K2 result changes to push' }
git commit -m 'S100 K2 V18 oracle results: exact but below S100 gate'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-s100-k2-oracle-v18
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'K2 V18 results pushed.' -ForegroundColor Green
