$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Scripts = @(
    (Join-Path $Repo 'pro_research\s100_mtp_inventory.py'),
    (Join-Path $Repo 'pro_research\s100_kverify_state_budget.py')
)
foreach ($s in $Scripts) { if (-not (Test-Path $s)) { throw "Missing metadata runner: $s" } }
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
Write-Host "Repository : $Repo"
Write-Host 'Mode       : CPU/header-only; no CUDA context or tensor payload reads'
foreach ($s in $Scripts) {
    Write-Host ''
    Write-Host ("=== " + (Split-Path -Leaf $s) + " ===") -ForegroundColor Cyan
    $old = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python $s
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "S100 metadata runner failed with native exit code $code: $s" }
}
Write-Host ''
Write-Host 'S100 metadata inventory completed without CUDA.' -ForegroundColor Green
