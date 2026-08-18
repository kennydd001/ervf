
param(
    [ValidateSet('qfast','mamba','fast','k5','k4','fast_k5','fast_k4','all')]
    [string]$Profile = 'all',
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing runtime environment: $Py" }

$Profiles = if ($Profile -eq 'all') {
    @('qfast','mamba','fast','k5','k4','fast_k5','fast_k4')
} else {
    @($Profile)
}
$Roles = @('exact_a','cand_a','cand_b','exact_b')

foreach ($P in $Profiles) {
    Write-Host ''
    Write-Host "=== phase 4 fresh timing [$P / $Mode] ===" -ForegroundColor Cyan
    foreach ($Role in $Roles) {
        & $Py 'pro_research\s100_phase4_fresh_arm.py' `
            --profile $P --role $Role --mode $Mode
        if ($LASTEXITCODE -ne 0) {
            throw "Fresh timing arm $P/$Mode/$Role returned $LASTEXITCODE"
        }
    }
    & $Py 'pro_research\s100_phase4_fresh_compare.py' `
        --profile $P --mode $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "Fresh comparison $P/$Mode returned $LASTEXITCODE"
    }
}
