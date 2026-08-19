param(
 [string]$Repo='',
 [string]$ModelDir='',
 [switch]$SkipSubsetSearch,
 [switch]$SkipPublish
)

$ErrorActionPreference='Stop'
if($PSVersionTable.PSVersion.Major -ge 7){
 $global:PSNativeCommandUseErrorActionPreference=$false
}

function Test-Repo([string]$Path){
 if(-not $Path -or -not(Test-Path $Path)){return $false}
 foreach($Relative in @(
  '.venv-nemotron\Scripts\python.exe',
  'pro_research\s100_phase10a_runtime.py',
  'pro_research\s100_lightning16_common.py',
  'pro_research\s100_lightning16_native.py',
  'pro_research\results\s100_lightning16\S100_LIGHTNING16_LAYER_SCREEN.json',
  'pro_research\results\s100_lightning16\S100_LIGHTNING16_QUALITY_SELECTION.json',
  'pro_research\results\s100_lightning16\S100_LIGHTNING16_BLOCK_VERIFIER.json',
  'pro_research\results\s100_lightning16\S100_LIGHTNING16_DFLASH2_ECONOMICS.json',
  'pro_research\results\s100_lightning15\S100_LIGHTNING15_TRACE_CALIBRATION.npz',
  'pro_research\results\s100_lightning15\S100_LIGHTNING15_TRACE_VALIDATION.npz',
  'pro_research\results\s100_lightning15\S100_LIGHTNING15_TRACE_HELDOUT.npz'
  )){
  if(-not(Test-Path (Join-Path $Path $Relative))){
   return $false
  }
 }
 return $true
}

function Find-Repo{
 if($Repo){
  $Resolved=(Resolve-Path $Repo).Path
  if(-not(Test-Repo $Resolved)){
   throw "Invalid completed Phase-16 repo: $Resolved"
  }
  return $Resolved
 }
 if(Test-Repo (Get-Location).Path){
  return (Get-Location).Path
 }
 $Root=Join-Path $env:USERPROFILE 'Documents\ChatGPT'
 $Candidates=@()
 foreach($Name in @('New project','ervf')){
  $Path=Join-Path $Root $Name
  if(Test-Path $Path){$Candidates += Get-Item $Path}
 }
 $Candidates += @(
  Get-ChildItem $Root -Directory `
   -Filter 'ervf-s100-reboot-*' -ErrorAction SilentlyContinue |
   Sort-Object LastWriteTime -Descending
 )
 foreach($Candidate in $Candidates){
  if(Test-Repo $Candidate.FullName){
   return $Candidate.FullName
  }
 }
 throw 'Completed Lightning Phase-16 worktree not found.'
}

function Test-LightningModel([string]$Path){
 if(-not $Path){return $false}
 $Config=Join-Path $Path 'config.json'
 $Index=Join-Path $Path 'model.safetensors.index.json'
 if(-not(Test-Path $Config) -or -not(Test-Path $Index)){
  return $false
 }
 try{
  $Object=Get-Content $Config -Raw|ConvertFrom-Json
  return [int64]$Object.max_position_embeddings -eq 1048576
 }catch{
  return $false
 }
}

function Find-Model([string]$Worktree){
 if($ModelDir){
  $Resolved=(Resolve-Path $ModelDir).Path
  if(-not(Test-LightningModel $Resolved)){
   throw "Model is not Lightning: $Resolved"
  }
  return $Resolved
 }
 if(
  $env:LS_MODEL_DIR -and
  (Test-LightningModel $env:LS_MODEL_DIR)
 ){
  return (Resolve-Path $env:LS_MODEL_DIR).Path
 }
 $Root=Join-Path $env:USERPROFILE 'Documents\ChatGPT'
 foreach($Candidate in @(
  (Join-Path $Worktree 'models\nemotron_3_5_lightning'),
  (Join-Path $Root 'New project\models\nemotron_3_5_lightning'),
  (Join-Path $Root 'ervf\models\nemotron_3_5_lightning')
 )){
  if(Test-LightningModel $Candidate){
   return (Resolve-Path $Candidate).Path
  }
 }
 foreach($Directory in Get-ChildItem $Root -Directory `
  -Filter 'ervf-s100-reboot-*' -ErrorAction SilentlyContinue){
  $Candidate=Join-Path $Directory.FullName `
   'models\nemotron_3_5_lightning'
  if(Test-LightningModel $Candidate){
   return (Resolve-Path $Candidate).Path
  }
 }

 # Hugging Face cache layout used by the measured Phase-16 run.
 $SearchRoots=@($Worktree)
 foreach($Name in @('New project','ervf')){
  $CandidateRoot=Join-Path $Root $Name
  if(Test-Path $CandidateRoot){$SearchRoots += $CandidateRoot}
 }
 $SearchRoots += @(
  Get-ChildItem $Root -Directory `
   -Filter 'ervf-s100-reboot-*' -ErrorAction SilentlyContinue |
   ForEach-Object {$_.FullName}
 )
 foreach($SearchRoot in @($SearchRoots|Sort-Object -Unique)){
  $Snapshots=Join-Path $SearchRoot (
   '.cache\nemotron_3_5_lightning\hub\' +
   'models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4\' +
   'snapshots'
  )
  if(-not(Test-Path $Snapshots)){continue}
  foreach($Snapshot in Get-ChildItem $Snapshots -Directory `
   -ErrorAction SilentlyContinue |
   Sort-Object LastWriteTime -Descending){
   if(Test-LightningModel $Snapshot.FullName){
    return $Snapshot.FullName
   }
  }
 }
 throw 'Lightning checkpoint not found.'
}

function Invoke-PythonStep(
 [string]$StepLabel,
 [string[]]$Arguments
){
 Write-Host ''
 Write-Host "=== $StepLabel ===" -ForegroundColor Cyan
 try{
  & $script:Python @Arguments
  $ExitCode=$LASTEXITCODE
  if($ExitCode){
   $script:Failures += "$StepLabel exit=$ExitCode"
   Write-Warning "$StepLabel exit=$ExitCode"
   return $false
  }
  return $true
 }catch{
  $script:Failures += "$StepLabel exception=$_"
  Write-Warning "$StepLabel failed: $_"
  return $false
 }
}

function Convert-ToSlug([string]$Value){
 return (($Value -replace '[^A-Za-z0-9]','_').Trim('_')).ToUpper()
}

function Add-CandidatesFromFile(
 [string]$Path,
 [hashtable]$Destination
){
 if(-not(Test-Path $Path)){return}
 $Document=Get-Content $Path -Raw|ConvertFrom-Json
 foreach($Candidate in @($Document.selected_for_validation)){
  if($null -eq $Candidate){continue}
  $CandidateName=[string]$Candidate.name
  if(-not $Destination.ContainsKey($CandidateName)){
   $Destination[$CandidateName]=$Candidate
  }
 }
}

$Worktree=Find-Repo
$Model=Find-Model $Worktree
$Python=Join-Path $Worktree '.venv-nemotron\Scripts\python.exe'
$script:Python=$Python
$env:LS_MODEL_DIR=$Model
Set-Location $Worktree

& $Python -c "import torch,cupy,moe_lab" 2>$null
if($LASTEXITCODE){
 & $Python -m pip install -e . --no-deps
 if($LASTEXITCODE){throw 'editable install failed'}
}

# Copy only Phase-16R files. Phase-16 evidence remains immutable.
if((Resolve-Path $PSScriptRoot).Path -ne $Worktree){
 Copy-Item `
  (Join-Path $PSScriptRoot 'pro_research\s100_lightning16r_*.py') `
  (Join-Path $Worktree 'pro_research') -Force
 Copy-Item `
  (Join-Path $PSScriptRoot 'pro_research\S100_LIGHTNING_PHASE16R_PREREGISTRATION.md') `
  (Join-Path $Worktree 'pro_research\S100_LIGHTNING_PHASE16R_PREREGISTRATION.md') `
  -Force
 Copy-Item `
  (Join-Path $PSScriptRoot 'agents\S100_LIGHTNING_PHASE16R_*.md') `
  (Join-Path $Worktree 'agents') `
  -Force
 Copy-Item `
  (Join-Path $PSScriptRoot 'RUN_S100_LIGHTNING_PHASE16R.ps1') `
  (Join-Path $Worktree 'RUN_S100_LIGHTNING_PHASE16R.ps1') `
  -Force
 Copy-Item `
  (Join-Path $PSScriptRoot 'PUBLISH_S100_LIGHTNING16R_GITHUB.ps1') `
  (Join-Path $Worktree 'PUBLISH_S100_LIGHTNING16R_GITHUB.ps1') `
  -Force
}

$Results=Join-Path $Worktree `
 'pro_research\results\s100_lightning16r'
$Logs=Join-Path $Worktree 'pro_research\results\logs'
$Stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
if(Test-Path $Results){
 $Existing=@(
  Get-ChildItem $Results -File `
   -Filter 'S100_LIGHTNING16R_*' `
   -ErrorAction SilentlyContinue
 )
 if($Existing.Count -gt 0){
  $Archive=Join-Path $Results "archive_$Stamp"
  New-Item -ItemType Directory -Force $Archive|Out-Null
  foreach($File in $Existing){
   Move-Item $File.FullName $Archive -Force
  }
 }
}
New-Item -ItemType Directory -Force $Results|Out-Null
New-Item -ItemType Directory -Force $Logs|Out-Null
$Log=Join-Path $Logs "S100_LIGHTNING16R_$Stamp.log"
$Failures=@()
Start-Transcript -Path $Log -Force|Out-Null

try{
 Write-Host "Worktree : $Worktree"
 Write-Host "Lightning: $Model"

 Invoke-PythonStep `
  '16R RECOVER PHASE-16 CALIBRATION IDENTITY' `
  @('pro_research\s100_lightning16r_recover.py') |
  Out-Null

 if(-not $SkipSubsetSearch){
  Invoke-PythonStep `
   '16R CALIBRATION-ONLY MAXIMAL K/V SUBSET SEARCH' `
   @('pro_research\s100_lightning16r_subset_search.py') |
   Out-Null
 }

 $Candidates=@{}
 Add-CandidatesFromFile `
  (Join-Path $Results 'S100_LIGHTNING16R_RECOVERY.json') `
  $Candidates
 Add-CandidatesFromFile `
  (Join-Path $Results 'S100_LIGHTNING16R_SUBSET_SEARCH.json') `
  $Candidates

 foreach($CandidateName in @($Candidates.Keys | Sort-Object)){
  $Candidate=$Candidates[$CandidateName]
  $Terms=[string][int]$Candidate.terms
  $Handoff=[string]$Candidate.handoff
  $Cases=@($Candidate.cases)
  $CasesJson=ConvertTo-Json -InputObject $Cases -Compress
  $Slug=Convert-ToSlug $CandidateName

  $CalibrationArguments=@(
   'pro_research\s100_lightning16r_quality.py',
   '--name',$CandidateName,
   '--terms',$Terms,
   '--cases-json',$CasesJson,
   '--split','calibration',
   '--handoff',$Handoff
  )
  $CalibrationOk=Invoke-PythonStep `
   "16R CLEAN CALIBRATION $CandidateName" `
   $CalibrationArguments
  if(-not $CalibrationOk){continue}
  $CalibrationPath=Join-Path $Results `
   "S100_LIGHTNING16R_QUALITY_${Slug}_CALIBRATION.json"
  if(-not(Test-Path $CalibrationPath)){continue}
  $Calibration=Get-Content $CalibrationPath -Raw|ConvertFrom-Json
  if(-not $Calibration.strict_pass){continue}

  $ValidationArguments=@(
   'pro_research\s100_lightning16r_quality.py',
   '--name',$CandidateName,
   '--terms',$Terms,
   '--cases-json',$CasesJson,
   '--split','validation',
   '--handoff',$Handoff
  )
  $ValidationOk=Invoke-PythonStep `
   "16R VALIDATION $CandidateName" `
   $ValidationArguments

  if(-not $ValidationOk){continue}
  $ValidationPath=Join-Path $Results `
   "S100_LIGHTNING16R_QUALITY_${Slug}_VALIDATION.json"
  if(-not(Test-Path $ValidationPath)){continue}
  $Validation=Get-Content $ValidationPath -Raw|ConvertFrom-Json
  if(-not $Validation.strict_pass){continue}

  $HeldoutArguments=@(
   'pro_research\s100_lightning16r_quality.py',
   '--name',$CandidateName,
   '--terms',$Terms,
   '--cases-json',$CasesJson,
   '--split','heldout',
   '--handoff',$Handoff
  )
  Invoke-PythonStep `
   "16R HELDOUT $CandidateName" `
   $HeldoutArguments |
   Out-Null
 }

 Invoke-PythonStep `
  '16R BRACKETED GRAPH-PARENT THROUGHPUT' `
  @('pro_research\s100_lightning16r_throughput.py') |
  Out-Null

 Invoke-PythonStep `
  '16R FINAL SUMMARY' `
  @('pro_research\s100_lightning16r_summary.py') |
  Out-Null

 $SummaryText=Join-Path $Results `
  'S100_LIGHTNING16R_SUMMARY.txt'
 if(Test-Path $SummaryText){
  Get-Content $SummaryText
 }

 if(-not $SkipPublish){
  try{
   & (Join-Path $PSScriptRoot `
    'PUBLISH_S100_LIGHTNING16R_GITHUB.ps1')
  }catch{
   Write-Warning "GitHub publication failed: $_"
  }
 }

 Write-Host ''
 Write-Host "Non-fatal technical failures: $($Failures.Count)"
 foreach($Failure in $Failures){
  Write-Host " - $Failure"
 }
 Write-Host "Transcript: $Log"
 Write-Host (
  'Return RECOVERY, SUBSET_SEARCH, all QUALITY files, ' +
  'THROUGHPUT and SUMMARY.'
 ) -ForegroundColor Green
}finally{
 Stop-Transcript|Out-Null
}
