param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\e100_mrhs256_v3.py'
$Verifier = Join-Path $Repo 'pro_research\verify_e100_mrhs256_v3.py'
$OutDir = Join-Path $Repo 'pro_research\results\e100_mrhs256'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }
if (-not (Test-Path $Verifier)) { throw "Verifier not found: $Verifier" }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("E100_MRHS256_${Mode}_${Stamp}.console.log")

function Invoke-NativePython {
    param([string[]]$Arguments, [switch]$Append)
    $old = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($Append) { & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append }
        else { & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "Python returned native exit code $code: $($Arguments -join ' ')" }
}

Write-Host "Repository : $Repo"
Write-Host "Branch     : pro-e100-batch"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $Log"
Write-Host ''
Write-Host '=== E100 full-warp MRHS256 vs adopted V6 selective baseline ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Runner, '--mode', $Mode)
Write-Host ''
Write-Host '=== Independent V3 verifier ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Verifier) -Append
Write-Host ''
Write-Host 'E100-MRHS256 V3 run and independent verification completed.' -ForegroundColor Green
