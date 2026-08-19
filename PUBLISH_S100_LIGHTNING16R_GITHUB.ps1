param(
 [string]$RemoteBranch=
  'agent/s100-lightning-phase16r-recovery-hardware'
)

$ErrorActionPreference='Stop'
$Repository=(& git rev-parse --show-toplevel 2>$null).Trim()
if(-not $Repository){throw 'Not inside a git worktree'}
Set-Location $Repository

$Paths=@()
foreach($Pattern in @(
 'RUN_S100_LIGHTNING_PHASE16R.ps1',
 'PUBLISH_S100_LIGHTNING16R_GITHUB.ps1',
 'agents\S100_LIGHTNING_PHASE16R_*.md',
 'pro_research\S100_LIGHTNING_PHASE16R_*.md',
 'pro_research\s100_lightning16r_*.py',
 'pro_research\results\s100_lightning16r\*.json',
 'pro_research\results\s100_lightning16r\*.txt'
)){
 $Paths += Get-ChildItem $Pattern -File `
  -ErrorAction SilentlyContinue |
  ForEach-Object {$_.FullName}
}
$Paths=$Paths|Sort-Object -Unique
if($Paths.Count -eq 0){
 throw 'No Phase-16R files found'
}

$AlreadyStaged=@(git diff --cached --name-only)
if($LASTEXITCODE){throw 'staged-file inspection failed'}
foreach($Item in $AlreadyStaged){
 $Normalized=$Item.Replace('\','/')
 if(
  $Normalized -ne 'RUN_S100_LIGHTNING_PHASE16R.ps1' -and
  $Normalized -ne 'PUBLISH_S100_LIGHTNING16R_GITHUB.ps1' -and
  -not $Normalized.StartsWith(
   'agents/S100_LIGHTNING_PHASE16R_'
  ) -and
  -not $Normalized.StartsWith(
   'pro_research/S100_LIGHTNING_PHASE16R_'
  ) -and
  -not $Normalized.StartsWith(
   'pro_research/s100_lightning16r_'
  ) -and
  -not $Normalized.StartsWith(
   'pro_research/results/s100_lightning16r/'
  )
 ){
  throw "Unrelated staged file present: $Item"
 }
}

git add -- $Paths
if($LASTEXITCODE){throw 'git add failed'}
git diff --cached --quiet
$DiffExit=$LASTEXITCODE
if($DiffExit -gt 1){throw "git diff failed: $DiffExit"}
if($DiffExit -eq 1){
 $CommitMessage=(
  'research: recover Lightning Phase 16 native selection ' +
  'and benchmark graph-parent speed'
 )
 git commit -m $CommitMessage
 if($LASTEXITCODE){throw 'git commit failed'}
}
git push origin "HEAD:refs/heads/$RemoteBranch"
if($LASTEXITCODE){throw 'git push failed'}
Write-Host "Published $RemoteBranch" -ForegroundColor Green
