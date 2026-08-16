$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$Path = 'pro_research/results/v12_async'
if (-not (Test-Path $Path)) { throw "No V12 results found at $Path" }
git add -f $Path
git commit -m "PRO V12 async harvest results"
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-v12-async
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'V12 results pushed to pro-v12-async.' -ForegroundColor Green
