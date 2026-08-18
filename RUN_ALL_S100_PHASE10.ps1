# S100 phase 10 one-click master: runs 10A (panel cache) then 10B (Mamba
# bandwidth) sequentially. Never in parallel — the GPU timings would
# contaminate each other. Exit 0 only when both sub-runners exit 0.
param([string]$Repo='')
$ErrorActionPreference='Stop'
$here=$PSScriptRoot
$a=Join-Path $here 'RUN_ALL_S100_PHASE10A.ps1'
$b=Join-Path $here 'RUN_ALL_S100_PHASE10B.ps1'
if($Repo){& $a -Repo $Repo}else{& $a}
if($LASTEXITCODE){throw '10A failed'}
if($Repo){& $b -Repo $Repo}else{& $b}
if($LASTEXITCODE){throw '10B failed'}
Write-Host 'S100 PHASE 10 COMPLETE — see pro_research\results\s100_phase10a and s100_phase10b'
