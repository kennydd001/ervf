$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\s100_kverify_mamba_rollback.py'
$Verifier = Join-Path $Repo 'pro_research\verify_s100_kverify_k1.py'
$OutDir = Join-Path $Repo 'pro_research\results\s100_kverify'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
foreach ($p in @($Runner,$Verifier)) { if (-not (Test-Path $p)) { throw "Missing K1 file: $p" } }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("S100_K1_${Stamp}.console.log")
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
        throw "Python returned native exit code $code: $($Arguments -join ' ')"
    }
}

Write-Host "Repository : $Repo"
Write-Host 'Experiment : S100-KVERIFY K1 exact Mamba rollback proof'
Write-Host "Log        : $Log"
Write-Host ''
Invoke-NativePython -Arguments @($Runner) -AllowScientificFailure
$runnerCode = $script:LastNativeCode
Write-Host ''
Write-Host '=== Independent verifier (runs even on negative rollback status) ===' -ForegroundColor Cyan
Invoke-NativePython -Arguments @($Verifier) -Append
Write-Host ''
if ($runnerCode -ne 0) {
    Write-Warning "K1 runner returned scientific/technical failure code $runnerCode after evidence was independently verified."
    exit $runnerCode
}
Write-Host 'S100 K1 rollback proof and independent verification completed.' -ForegroundColor Green
