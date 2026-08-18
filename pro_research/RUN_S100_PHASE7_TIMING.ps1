
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
$Heldout = Get-Content `
    'pro_research\results\S100_PHASE7_HELDOUT.json' `
    -Raw | ConvertFrom-Json

$Failures = @()
foreach ($Property in $Heldout.results.PSObject.Properties) {
    if ($Property.Value.status -ne 'v18_fidelity_candidate') { continue }
    $Candidate = $Property.Name
    $Ok = $true
    foreach ($Role in @(
        'base_a','legacy_cand','cand_a','cand_b','base_b'
    )) {
        & $Py 'pro_research\s100_phase7_candidate_arm.py' `
            --candidate $Candidate --role $Role
        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Candidate/$Role"
            $Ok = $false
            break
        }
    }
    if ($Ok) {
        & $Py 'pro_research\s100_phase7_candidate_compare.py' `
            --candidate $Candidate
        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Candidate/compare"
        }
    }
}
if ($Failures.Count -gt 0) {
    Write-Warning ("Timing failures: " + ($Failures -join ', '))
}
exit 0
