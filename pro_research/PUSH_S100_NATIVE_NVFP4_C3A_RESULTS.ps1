$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') { throw "Wrong branch: $Branch" }
$Result = 'pro_research/results/native_nvfp4/C3A_REAL_WEIGHT.json'
if (-not (Test-Path $Result)) { throw "Missing $Result" }
Write-Host 'Committing and pushing ONLY the C3A result path:'
git status --short -- $Result
git add -- $Result
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
git diff --cached --stat -- $Result
git commit -m 'C3A: record real checkpoint native FP4 result' -- $Result
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin $Branch
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
git rev-parse HEAD
