$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2') { throw "Expected pro-s100-nativefp4-c2, got '$Branch'" }
$Path = 'pro_research/results/native_nvfp4/C2_TORCH_CONTRACT.json'
if (-not (Test-Path $Path)) { throw "Missing C2 result: $Path" }
git add -f -- $Path
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --cached --check failed' }
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No C2 result changes to push' }
git commit -m 'S100 native NVFP4 C2 Torch execution results'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-s100-nativefp4-c2
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'C2 native NVFP4 result pushed.' -ForegroundColor Green
