param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\e100_nvfp4_tiled_mrhs.py'
$Verifier = Join-Path $Repo 'pro_research\verify_e100_nvfp4_tiled_mrhs.py'
$ResultDir = Join-Path $Repo 'pro_research\results\e100_nvfp4_tiled_mrhs'
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $ResultDir "E100_NVFP4_TILED_MRHS_${Mode}_${Stamp}.console.log"

if ((git branch --show-current).Trim() -ne 'pro-e100-batch') {
    throw "Refusing to run outside pro-e100-batch"
}
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

function Invoke-NativePython {
    param([string[]]$Arguments, [switch]$Append, [switch]$AllowScientificFailure)
    $old = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($Append) { & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append }
        else { & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log }
        $code = $LASTEXITCODE
        $script:LastNativeCode = $code
    }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0 -and -not $AllowScientificFailure) {
        throw "Python returned native exit code ${code}: $($Arguments -join ' ')"
    }
}

Write-Host "Repository : $Repo"
Write-Host "Branch     : pro-e100-batch"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $Log"
Write-Host ''
Write-Host '=== Exact NVFP4 tiled shared-decode MRHS V5 ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Runner, '--mode', $Mode) -AllowScientificFailure
$runnerCode = $script:LastNativeCode
Write-Host ''
Write-Host '=== Independent verifier ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Verifier) -Append -AllowScientificFailure
$verifyCode = $script:LastNativeCode

if ($runnerCode -ne 0 -or $verifyCode -ne 0) {
    throw "Tiled MRHS V5 failed closed: runner=${runnerCode}, verifier=${verifyCode}. Inspect raw JSON/log before interpretation."
}
Write-Host ''
Write-Host 'E100 tiled NVFP4 MRHS run and verification completed.' -ForegroundColor Green
