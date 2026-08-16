param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke',
    [switch]$IncludeAddNormDiagnostic
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$MasterLog = Join-Path $OutDir ("E50_LADDER_${Mode}_${Stamp}.console.log")
$Failures = @()

function Run-Step([string]$Name, [string]$Script) {
    Write-Host ''
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    try {
        & $Script -Mode $Mode 2>&1 | Tee-Object -FilePath $MasterLog -Append
        if ($LASTEXITCODE -ne 0) { throw "$Name returned code $LASTEXITCODE" }
    }
    catch {
        $Failures += "$Name :: $($_.Exception.Message)"
        Write-Warning $Failures[-1]
    }
}

Write-Host "Repository : $Repo"
Write-Host "Mode       : $Mode"
Write-Host "Master log : $MasterLog"

Run-Step 'V12 fixed-K queue/event baseline' (Join-Path $Repo 'pro_research\RUN_V12.ps1')
Run-Step 'V12B rolling busy-query credit' (Join-Path $Repo 'pro_research\RUN_V12B.ps1')
Run-Step 'V12C rolling blocking-event credit' (Join-Path $Repo 'pro_research\RUN_V12C.ps1')

if ($IncludeAddNormDiagnostic) {
    Write-Host ''
    Write-Host '=== AddNorm late-divergence diagnostic ===' -ForegroundColor Cyan
    try {
        & (Join-Path $Repo 'pro_research\RUN_ADDNORM_DIAG.ps1') -Tokens 160 2>&1 | Tee-Object -FilePath $MasterLog -Append
        if ($LASTEXITCODE -ne 0) { throw "AddNorm diagnostic returned code $LASTEXITCODE" }
    }
    catch {
        $Failures += "AddNorm diagnostic :: $($_.Exception.Message)"
        Write-Warning $Failures[-1]
    }
}

Write-Host ''
Write-Host '=== Build conservative E50 report ===' -ForegroundColor Cyan
& $Python (Join-Path $Repo 'pro_research\build_v12_e50_report.py') 2>&1 | Tee-Object -FilePath $MasterLog -Append
if ($LASTEXITCODE -ne 0) { $Failures += "report builder returned code $LASTEXITCODE" }

Write-Host ''
if ($Failures.Count -eq 0) {
    Write-Host 'E50 ladder completed without technical failures.' -ForegroundColor Green
    Write-Host 'Push with: .\pro_research\PUSH_V12_RESULTS.ps1'
    exit 0
}
Write-Warning ("E50 ladder completed with technical failures:`n - " + ($Failures -join "`n - "))
Write-Host 'Successful independent subarms were preserved in pro_research\results\v12_async.'
exit 2
