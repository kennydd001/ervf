
param(
    [ValidateSet('qfast','mamba','k5','k4','fast','fast_k5','fast_k4','all')]
    [string]$Profile = 'all',
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke',
    [switch]$ForceTrace
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing runtime environment: $Py" }

# First prove that the existing control and FAST result are internally
# consistent. This is CPU-only and repairs the phase-3 serialization failure.
foreach ($Existing in @(
    @{ Profile='k1_control'; Mode='smoke' },
    @{ Profile='fast'; Mode='smoke' }
)) {
    $P = "pro_research\results\S100_PHASE3_FIDELITY_$($Existing.Profile.ToUpper())_$($Existing.Mode.ToUpper()).json"
    if (Test-Path $P) {
        & $Py 'pro_research\verify_s100_phase3_fidelity.py' `
            --profile $Existing.Profile --mode $Existing.Mode
        if ($LASTEXITCODE -ne 0) {
            throw "Existing fidelity result failed verification: $P"
        }
    }
}

$Profiles = if ($Profile -eq 'all') {
    @('qfast','mamba','k5','k4')
} else {
    @($Profile)
}

foreach ($P in $Profiles) {
    if ($Mode -eq 'full') {
        $Smoke = "pro_research\results\S100_PHASE3_FIDELITY_$($P.ToUpper())_SMOKE.json"
        if (-not (Test-Path $Smoke)) {
            throw "Missing smoke result for $P; run smoke first."
        }
        $S = Get-Content $Smoke -Raw | ConvertFrom-Json
        if ($S.status -ne 'v18_fidelity_candidate') {
            Write-Warning "Skipping $P/full because smoke status is $($S.status)"
            continue
        }
    }

    $Args = @(
        '-ExecutionPolicy','Bypass','-File',
        'pro_research\RUN_S100_PHASE3_FIDELITY.ps1',
        '-Profile',$P,'-Mode',$Mode,'-SkipControl'
    )
    if ($ForceTrace) { $Args += '-ForceTrace' }
    & powershell @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Fidelity $P/$Mode returned $LASTEXITCODE"
    }
}
