$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4') { throw "Refusing push from branch '$Branch'; expected pro-s100-nativefp4" }
$Path = 'pro_research/results/native_nvfp4'
if (-not (Test-Path $Path)) { throw "No native NVFP4 result directory found: $Path" }
git add -f -- $Path
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No staged native NVFP4 result changes.' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --cached --check failed' }
git commit -m 'S100 native NVFP4 capability results'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-s100-nativefp4
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'Native NVFP4 results pushed to pro-s100-nativefp4.' -ForegroundColor Green
