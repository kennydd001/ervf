param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = '.\.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    throw "Missing runtime Python: $Py"
}

$Budgets = @(64, 128, 192, 256, 320)
$Failures = @()

foreach ($Budget in $Budgets) {
    $Roles = @('base_a', 'cand_a', 'cand_b', 'base_b')
    if ($Mode -eq 'smoke') {
        $Roles += 'bad'
    }

    $BudgetOk = $true
    foreach ($Role in $Roles) {
        & $Py 'pro_research\s100_phase8_backend_arm.py' `
            --budget $Budget `
            --mode $Mode `
            --role $Role

        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Budget/$Mode/$Role"
            $BudgetOk = $false
            break
        }
    }

    if ($BudgetOk) {
        & $Py 'pro_research\s100_phase8_backend_compare.py' `
            --budget $Budget `
            --mode $Mode

        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Budget/$Mode/compare"
        }
    }
}

if ($Mode -eq 'full') {
    & $Py 'pro_research\s100_phase8_backend_select.py'
    if ($LASTEXITCODE -ne 0) {
        $Failures += 'backend_select'
    }
}

if ($Failures.Count -gt 0) {
    $FailureText = $Failures -join ', '
    Write-Warning "Phase-8 per-budget failures: $FailureText"
}

exit 0
