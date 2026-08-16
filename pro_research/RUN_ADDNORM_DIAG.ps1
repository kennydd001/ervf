param([int]$Tokens = 160)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\diag_addnorm_late_divergence.py'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("ADDNORM_DIAG_${Stamp}.console.log")
& $Python $Runner --tokens $Tokens 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "AddNorm diagnostic returned code $LASTEXITCODE" }
