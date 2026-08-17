$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
$Heldout = Get-Content 'pro_research\results\S100_PHASE6_HELDOUT.json' -Raw | ConvertFrom-Json
$Failures = @()

foreach ($Property in $Heldout.results.PSObject.Properties) {
    if ($Property.Value.status -ne 'v18_fidelity_candidate') { continue }
    $Name = $Property.Name
    $Ok = $true
    foreach ($Role in @('base_a','cand_a','cand_b','base_b')) {
        & $Py 'pro_research\s100_phase6_candidate_arm.py' `
            --candidate $Name --role $Role
        if ($LASTEXITCODE -ne 0) {
            $Failures += "$Name/$Role"
            $Ok = $false
            break
        }
    }
    if ($Ok) {
        & $Py 'pro_research\s100_phase6_candidate_compare.py' --candidate $Name
        if ($LASTEXITCODE -ne 0) { $Failures += "$Name/compare" }
    }
}
if ($Failures.Count -gt 0) {
    Write-Warning ("Candidate timing failures: " + ($Failures -join ', '))
}
exit 0
