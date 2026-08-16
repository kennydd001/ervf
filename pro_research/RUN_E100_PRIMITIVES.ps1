param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$Steps = @(
    @{ Name = 'MRHS32 common-weight N2/N4/N8'; Script = '.\pro_research\RUN_E100_MRHS.ps1' },
    @{ Name = 'MRHS256 common-weight N4/N8/N16'; Script = '.\pro_research\RUN_E100_MRHS256.ps1' },
    @{ Name = 'PAIRBATCH routed N4 P=24'; Script = '.\pro_research\RUN_E100_PAIRBATCH.ps1' }
)
$Failures = @()

Write-Host "Repository : $Repo"
Write-Host "Branch     : pro-e100-batch"
Write-Host "Mode       : $Mode"
Write-Host ''
foreach ($Step in $Steps) {
    Write-Host ("=== " + $Step.Name + " ===") -ForegroundColor Cyan
    try {
        & $Step.Script -Mode $Mode
        if ($LASTEXITCODE -ne 0) { throw "native/script exit code $LASTEXITCODE" }
    }
    catch {
        $Failures += ($Step.Name + ' :: ' + $_.Exception.Message)
        Write-Warning $Failures[-1]
        if ($Mode -eq 'smoke') {
            Write-Warning 'Smoke ladder is fail-closed; later primitives are not run after a technical/correctness failure.'
            break
        }
    }
    Write-Host ''
}

if ($Failures.Count -gt 0) {
    Write-Warning ("E100 primitive ladder had failures:`n - " + ($Failures -join "`n - "))
    exit 2
}
Write-Host 'E100 primitive ladder completed. Interpret each preregistered result separately; do not multiply speedups.' -ForegroundColor Green
Write-Host 'Push evidence with: .\pro_research\PUSH_E100_RESULTS.ps1'
