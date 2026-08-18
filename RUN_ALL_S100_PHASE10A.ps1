param([string]$Repo='')
$ErrorActionPreference='Stop'
function Find-Repo{
 if($Repo){return (Resolve-Path $Repo).Path}
 $root=Join-Path $env:USERPROFILE 'Documents\ChatGPT'
 foreach($d in Get-ChildItem $root -Directory -Filter 'ervf-s100-reboot-*'|Sort-Object LastWriteTime -Descending){
  if(Test-Path (Join-Path $d.FullName 'agents\S100_PHASE9_FINDINGS.md')){return $d.FullName}
 }
 throw 'Completed Phase-9 worktree not found.'
}
$R=Find-Repo;$Py=Join-Path $R '.venv-nemotron\Scripts\python.exe';Set-Location $R
if((Resolve-Path $PSScriptRoot).Path -ne $R){
 Copy-Item (Join-Path $PSScriptRoot 'pro_research\*') 'pro_research' -Force
 Copy-Item (Join-Path $PSScriptRoot 'agents\S100_PHASE10A_PANEL_CACHE_PLAN.md') 'agents\S100_PHASE10A_PANEL_CACHE_PLAN.md' -Force
}
New-Item -ItemType Directory -Force 'pro_research\results\s100_phase10a'|Out-Null
& $Py 'pro_research\s100_phase10a_profile.py' --alpha 0.0003
if($LASTEXITCODE){throw '10A profile failed'}
$good=@()
foreach($b in @(8,16,24,32,40,48)){
 $ok=$true
 foreach($role in @('base_a','cand_a','cand_b','base_b','bad')){
  & $Py 'pro_research\s100_phase10a_arm.py' --budget $b --mode smoke --role $role
  if($LASTEXITCODE){$ok=$false;break}
 }
 if($ok){
  & $Py 'pro_research\s100_phase10a_compare.py' --budget $b --mode smoke
  $q=Get-Content "pro_research\results\s100_phase10a\P10A_COMPARE_${b}_smoke.json" -Raw|ConvertFrom-Json
  if($q.status -eq 'pass'){$good+=$b}
 }
}
foreach($b in $good){
 $ok=$true
 foreach($role in @('base_a','cand_a','cand_b','base_b')){
  & $Py 'pro_research\s100_phase10a_arm.py' --budget $b --mode full --role $role
  if($LASTEXITCODE){$ok=$false;break}
 }
 if($ok){& $Py 'pro_research\s100_phase10a_compare.py' --budget $b --mode full}
}
& $Py 'pro_research\s100_phase10a_select.py'
Get-Content 'pro_research\results\s100_phase10a\S100_PHASE10A_SUMMARY.txt'
