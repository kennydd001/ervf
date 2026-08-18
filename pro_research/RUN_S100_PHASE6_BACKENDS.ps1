param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
$Failures = @()

foreach ($Backend in @('ballot_fused','direct','direct_opt')) {
    $BackendOk = $true
    foreach ($Role in @('base_a','cand_a','cand_b','base_b')) {
        & $Py 'pro_research\s100_phase6_backend_arm.py' `
            --backend $Backend --role $Role --mode $Mode
        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Backend/$Mode/$Role"
            $BackendOk = $false
            break
        }
    }
    if ($BackendOk) {
        & $Py 'pro_research\s100_phase6_backend_compare.py' `
            --backend $Backend --mode $Mode
        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Backend/$Mode/compare"
        }
    }
}

if ($Mode -eq 'full') {
    & $Py 'pro_research\s100_phase6_backend_select.py'
    if ($LASTEXITCODE -ne 0) { $Failures += 'backend_select' }
}

if ($Failures.Count -gt 0) {
    Write-Warning ("Exact backend failures; quality path will continue with legacy if needed: " + ($Failures -join ', '))
}
exit 0
