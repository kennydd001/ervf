param(
 [Parameter(Mandatory=$true)][string]$Destination,
 [string]$Revision='0dcd680e5585c791728c83342b311d0a0026dbeb',
 [string]$Python=''
)
$ErrorActionPreference='Stop'
$Repo=(& git rev-parse --show-toplevel 2>$null).Trim()
if(-not $Repo){throw 'Run inside the ERVF worktree.'}
if(-not $Python){
 $Candidates=@(
  (Join-Path $Repo '.venv-nemotron\Scripts\python.exe'),
  (Join-Path $Repo '.venv\Scripts\python.exe'),
  'python'
 )
 foreach($C in $Candidates){
  if($C -eq 'python' -or (Test-Path $C)){$Python=$C;break}
 }
}
$Dest=[System.IO.Path]::GetFullPath($Destination)
Write-Host "Destination: $Dest" -ForegroundColor Cyan
Write-Host "Official model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

& $Python (Join-Path $Repo 'pro_research\lightning_migration\acquire_lightning.py') `
 --destination $Dest --revision $Revision
if($LASTEXITCODE){throw "Lightning acquisition failed: exit=$LASTEXITCODE"}

& $Python (Join-Path $Repo 'pro_research\lightning_migration\model_guard.py') `
 $Dest
if($LASTEXITCODE){throw "Downloaded model did not pass Lightning guard: exit=$LASTEXITCODE"}

Write-Host 'Lightning acquisition and identity gate passed.' -ForegroundColor Green
