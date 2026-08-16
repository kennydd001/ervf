$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent (Split-Path -Parent $Here)
$Rel = 'pro_research/results/pro_max_v2'

git -C $Repo add -f $Rel
$changes = git -C $Repo diff --cached --name-only
if (-not $changes) {
    Write-Host 'Geen nieuwe PRO-MAX V2-resultaten om te committen.'
    exit 0
}
git -C $Repo commit -m 'PRO-MAX V2 GPU results'
git -C $Repo push origin pro-max-v2
