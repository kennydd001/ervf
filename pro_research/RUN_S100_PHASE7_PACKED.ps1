
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'

$SmokeOk = $true
foreach ($Role in @('base_a','cand_a','cand_b','base_b','bad')) {
    & $Py 'pro_research\s100_phase7_backend_arm.py' `
        --mode smoke --role $Role
    if ($LASTEXITCODE -ne 0) {
        $SmokeOk = $false
        Write-Warning "Packed smoke role failed: $Role"
    }
}

if ($SmokeOk) {
    & $Py 'pro_research\s100_phase7_backend_compare.py' --mode smoke
    if ($LASTEXITCODE -ne 0) { $SmokeOk = $false }
}

if ($SmokeOk) {
    foreach ($Role in @('base_a','cand_a','cand_b','base_b')) {
        & $Py 'pro_research\s100_phase7_backend_arm.py' `
            --mode full --role $Role
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Packed full role failed: $Role"
            break
        }
    }
    & $Py 'pro_research\s100_phase7_backend_compare.py' --mode full
}

& $Py 'pro_research\s100_phase7_backend_select.py'
exit 0
