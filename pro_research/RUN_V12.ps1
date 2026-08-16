param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\queue_stream_v12.py'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("V12_${Mode}_${Stamp}.console.log")
Write-Host "Repository : $Repo"
Write-Host "Python     : $Python"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $Log"
& $Python $Runner --mode $Mode 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "V12 returned code $LASTEXITCODE" }
