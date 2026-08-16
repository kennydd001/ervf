param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\credit_stream_v12b_entry.py'
$Verifier = Join-Path $Repo 'pro_research\verify_v12b.py'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }
if (-not (Test-Path $Verifier)) { throw "Verifier not found: $Verifier" }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("V12B_${Mode}_${Stamp}.console.log")
Write-Host "Repository : $Repo"
Write-Host "Python     : $Python"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $Log"
Write-Host ''
Write-Host '=== V12B rolling credit-window run ==='
& $Python $Runner --mode $Mode 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "V12B returned code $LASTEXITCODE" }
Write-Host ''
Write-Host '=== V12B independent verifier ==='
& $Python $Verifier 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "V12B verifier returned code $LASTEXITCODE" }
