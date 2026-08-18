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
 Copy-Item (Join-Path $PSScriptRoot 'agents\S100_PHASE10B_MAMBA_BANDWIDTH_PLAN.md') 'agents\S100_PHASE10B_MAMBA_BANDWIDTH_PLAN.md' -Force
}
New-Item -ItemType Directory -Force 'pro_research\results\s100_phase10b'|Out-Null
& $Py 'pro_research\s100_phase10b_stream_bench.py'
if($LASTEXITCODE){throw '10B cold-stream benchmark failed'}
$s=Get-Content 'pro_research\results\s100_phase10b\S100_PHASE10B_STREAM.json' -Raw|ConvertFrom-Json
foreach($v in $s.selected_for_integration){
 $ok=$true
 foreach($role in @('base_a','cand_a','cand_b','base_b')){
  & $Py 'pro_research\s100_phase10b_arm.py' --variant $v --mode smoke --role $role
  if($LASTEXITCODE){$ok=$false;break}
 }
 if($ok){
  & $Py 'pro_research\s100_phase10b_compare.py' --variant $v --mode smoke
  $q=Get-Content "pro_research\results\s100_phase10b\P10B_COMPARE_${v}_smoke.json" -Raw|ConvertFrom-Json
  if($q.status -eq 'pass'){
   foreach($role in @('base_a','cand_a','cand_b','base_b')){
    & $Py 'pro_research\s100_phase10b_arm.py' --variant $v --mode full --role $role
    if($LASTEXITCODE){$ok=$false;break}
   }
   if($ok){& $Py 'pro_research\s100_phase10b_compare.py' --variant $v --mode full}
  }
 }
}
& $Py 'pro_research\s100_phase10b_select.py'
Get-Content 'pro_research\results\s100_phase10b\S100_PHASE10B_SUMMARY.txt'
