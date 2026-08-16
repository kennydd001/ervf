param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\e100_nvfp4_smem_mrhs.py'
$Verifier = Join-Path $Repo 'pro_research\verify_e100_nvfp4_smem_mrhs.py'
$OutDir = Join-Path $Repo 'pro_research\results\e100_nvfp4_smem_mrhs'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
foreach ($p in @($Runner,$Verifier)) { if (-not (Test-Path $p)) { throw "Missing E100 SMEM-MRHS file: $p" } }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("E100_NVFP4_SMEM_MRHS_${Mode}_${Stamp}.console.log")
$script:LastNativeCode = 0

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
Write-Host '=== Exact NVFP4 shared-decode MRHS ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Runner, '--mode', $Mode) -AllowScientificFailure
$runnerCode = $script:LastNativeCode
Write-Host ''
Write-Host '=== Independent verifier ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Verifier) -Append
Write-Host ''
if ($runnerCode -ne 0) {
    Write-Warning "SMEM-MRHS runner returned scientific/technical failure code $runnerCode after verification."
    exit $runnerCode
}
Write-Host 'E100 NVFP4 shared-decode MRHS completed.' -ForegroundColor Green
