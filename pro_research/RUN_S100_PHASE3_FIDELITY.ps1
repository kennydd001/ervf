param(
    [ValidateSet('qfast','mamba','fast','k5','k4','fast_k5','fast_k4','k1_control')]
    [string]$Profile = 'fast',
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke',
    [switch]$ForceTrace,
    [switch]$SkipControl
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing runtime environment: $Py" }
$TraceArgs = @('pro_research\s100_build_v18_fidelity_trace.py','--mode',$Mode)
if ($ForceTrace) { $TraceArgs += '--force' }
& $Py @TraceArgs
if ($LASTEXITCODE -ne 0) { throw "V18 trace builder returned $LASTEXITCODE" }
& $Py 'pro_research\verify_s100_phase3_trace.py' --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "V18 trace verifier returned $LASTEXITCODE" }
if (-not $SkipControl -and $Profile -ne 'k1_control') {
    $Control='pro_research\results\S100_PHASE3_FIDELITY_K1_CONTROL_SMOKE.json'; $Need=$true
    if (Test-Path $Control) { try { $C=Get-Content $Control -Raw|ConvertFrom-Json; $Need=$C.status -ne 'control_failed_as_expected' } catch { $Need=$true } }
    if ($Need) {
        & $Py 'pro_research\s100_build_v18_fidelity_trace.py' --mode smoke
        if ($LASTEXITCODE -ne 0) { throw 'Smoke trace for control failed' }
        & $Py 'pro_research\s100_phase3_fidelity.py' --profile k1_control --mode smoke
        if ($LASTEXITCODE -ne 0) { throw 'K1 control did not fail as expected' }
        & $Py 'pro_research\verify_s100_phase3_fidelity.py' --profile k1_control --mode smoke
        if ($LASTEXITCODE -ne 0) { throw 'K1 control verifier failed' }
    }
}
& $Py 'pro_research\s100_phase3_fidelity.py' --profile $Profile --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "Fidelity profile $Profile returned $LASTEXITCODE" }
& $Py 'pro_research\verify_s100_phase3_fidelity.py' --profile $Profile --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "Fidelity verifier for $Profile failed" }
