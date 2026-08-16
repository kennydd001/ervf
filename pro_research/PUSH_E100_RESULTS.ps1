$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$Path = 'pro_research/results/e100_mrhs'
if (-not (Test-Path $Path)) { throw "No E100-MRHS results found at $Path" }
$branch = (git branch --show-current).Trim()
if ($branch -ne 'pro-e100-batch') { throw "Refusing to push E100 results from branch '$branch'; expected pro-e100-batch" }
git add -f $Path
git commit -m "PRO E100 MRHS results"
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-e100-batch
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'E100-MRHS results pushed to pro-e100-batch.' -ForegroundColor Green
