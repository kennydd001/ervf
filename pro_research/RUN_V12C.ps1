param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\credit_wait_v12c_blocking_entry.py'
$Verifier = Join-Path $Repo 'pro_research\verify_v12c.py'
$BlockingVerifier = Join-Path $Repo 'pro_research\verify_v12c_blocking.py'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
foreach ($Path in @($Runner,$Verifier,$BlockingVerifier)) {
    if (-not (Test-Path $Path)) { throw "Required V12C file not found: $Path" }
}
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("V12C_${Mode}_${Stamp}.console.log")
Write-Host "Repository : $Repo"
Write-Host "Python     : $Python"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $Log"
Write-Host ''
Write-Host '=== V12C rolling blocking-event run ==='
& $Python $Runner --mode $Mode 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "V12C returned code $LASTEXITCODE" }
Write-Host ''
Write-Host '=== V12C independent numerical verifier ==='
& $Python $Verifier 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "V12C numerical verifier returned code $LASTEXITCODE" }
Write-Host ''
Write-Host '=== V12C blocking-event semantics verifier ==='
& $Python $BlockingVerifier 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "V12C blocking-event verifier returned code $LASTEXITCODE" }
