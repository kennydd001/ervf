$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$branch = (git branch --show-current).Trim()
if ($branch -ne 'pro-e100-batch') { throw "Refusing to push E100 results from branch '$branch'; expected pro-e100-batch" }
$Paths = @(
    'pro_research/results/e100_mrhs',
    'pro_research/results/e100_mrhs256',
    'pro_research/results/e100_pairbatch',
    'pro_research/results/e100_nvfp4_smem_mrhs'
) | Where-Object { Test-Path $_ }
if ($Paths.Count -eq 0) { throw 'No E100 primitive result directories found.' }
foreach ($Path in $Paths) { git add -f $Path }
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No staged E100 result changes to commit.' }
git commit -m "PRO E100 primitive results"
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-e100-batch
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'E100 primitive results pushed to pro-e100-batch.' -ForegroundColor Green
