
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = '.\.venv-nemotron\Scripts\python.exe'
$Candidates = @(
    'thr_0003','thr_0010','thr_0015','thr_0020',
    'k1','k2','thr0010_k1','thr0010_k2','thr0015_k1'
)
$Failures = @()
foreach ($Candidate in $Candidates) {
    & $Py 'pro_research\s100_phase7_heldout_arm.py' `
        --candidate $Candidate
    if ($LASTEXITCODE -ne 0) { $Failures += $Candidate }
}
& $Py 'pro_research\s100_phase7_heldout_collect.py'
if ($Failures.Count -gt 0) {
    Write-Warning ("Heldout process failures: " + ($Failures -join ', '))
}
exit 0
