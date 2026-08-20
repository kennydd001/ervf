param(
    [ValidateSet('compile', 'state', 'smoke', 'thermal', 'generalize', 'all')]
    [string]$Action = 'smoke',
    [string]$ModelDir = '',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
$Measure = Join-Path $Repo 'pro_research\s100_phase30e_measure.py'
$State = Join-Path $Repo 'pro_research\s100_phase30e_state.py'
$Adjudicate = Join-Path $Repo 'pro_research\s100_phase30e_adjudicate.py'
$Generalize = Join-Path $Repo 'pro_research\s100_phase30e_generalize.py'

function Test-LightningModel([string]$Path) {
    if (-not $Path) { return $false }
    $Config = Join-Path $Path 'config.json'
    $Index = Join-Path $Path 'model.safetensors.index.json'
    if (-not (Test-Path -LiteralPath $Config) -or
        -not (Test-Path -LiteralPath $Index)) { return $false }
    try {
        return [int64](
            Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        ).max_position_embeddings -eq 1048576
    } catch { return $false }
}

function Find-Python {
    if ($Python) {
        $Resolved = (Resolve-Path -LiteralPath $Python).Path
        return $Resolved
    }
    $Candidates = @(
        (Join-Path $Repo '.venv-nemotron\Scripts\python.exe'),
        (Join-Path (Split-Path -Parent $Repo) 'New project\.venv-nemotron\Scripts\python.exe')
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    }
    try { return (Get-Command python -ErrorAction Stop).Source }
    catch { throw 'Python environment not found.' }
}

function Find-LightningModel {
    if ($ModelDir) {
        $Resolved = (Resolve-Path -LiteralPath $ModelDir).Path
        if (-not (Test-LightningModel $Resolved)) {
            throw "Invalid Lightning checkpoint: $Resolved"
        }
        return $Resolved
    }
    if ($env:LS_MODEL_DIR -and (Test-LightningModel $env:LS_MODEL_DIR)) {
        return (Resolve-Path -LiteralPath $env:LS_MODEL_DIR).Path
    }
    $Roots = @($Repo, (Join-Path (Split-Path -Parent $Repo) 'New project'))
    foreach ($Root in $Roots) {
        $Direct = Join-Path $Root 'models\nemotron_3_5_lightning_v35'
        if (Test-LightningModel $Direct) { return $Direct }
        $Snapshots = Join-Path $Root (
            '.cache\nemotron_3_5_lightning\hub\' +
            'models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4\snapshots'
        )
        if (Test-Path -LiteralPath $Snapshots) {
            foreach ($Snapshot in Get-ChildItem -LiteralPath $Snapshots -Directory |
                Sort-Object LastWriteTime -Descending) {
                if (Test-LightningModel $Snapshot.FullName) {
                    return $Snapshot.FullName
                }
            }
        }
    }
    throw 'Nemotron 3.5 Lightning checkpoint not found.'
}

function Invoke-Step([string]$Label, [string[]]$Arguments) {
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $script:PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit $LASTEXITCODE" }
}

$script:PythonExe = Find-Python
$env:LS_MODEL_DIR = Find-LightningModel
Set-Location -LiteralPath $Repo

if ($Action -in @('compile', 'all')) {
    Invoke-Step 'Phase30E compile' @($Measure, '--arm', 'compile', '--tag', 'COMPILE')
    if ($Action -eq 'compile') { exit 0 }
}
if ($Action -in @('state', 'all')) {
    Invoke-Step 'Phase30E parent state' @($State, '--mode', 'parent')
    Invoke-Step 'Phase30E candidate state' @($State, '--mode', 'candidate')
    Invoke-Step 'Phase30E state compare' @($State, '--mode', 'compare')
    if ($Action -eq 'state') { exit 0 }
}
if ($Action -in @('smoke', 'all')) {
    foreach ($Arm in @('parent', 'combined', 'candidate')) {
        Invoke-Step "Phase30E smoke $Arm" @(
            $Measure, '--arm', $Arm, '--tag', 'SMOKE',
            '--blocks', '8', '--warmup', '4'
        )
    }
    if ($Action -eq 'smoke') { exit 0 }
}
if ($Action -in @('thermal', 'all')) {
    $Orders = @(
        @('parent', 'combined', 'candidate'),
        @('candidate', 'combined', 'parent'),
        @('combined', 'parent', 'candidate'),
        @('candidate', 'parent', 'combined')
    )
    for ($Round=1; $Round -le 4; $Round++) {
        $Tag = "R$Round"
        foreach ($Arm in $Orders[$Round-1]) {
            Invoke-Step "Phase30E $Tag $Arm" @(
                $Measure, '--arm', $Arm, '--tag', $Tag,
                '--blocks', '16', '--warmup', '8'
            )
        }
    }
    Invoke-Step 'Phase30E thermal adjudication' @(
        $Adjudicate, '--tags', 'R1', 'R2', 'R3', 'R4'
    )
    if ($Action -eq 'thermal') { exit 0 }
}
if ($Action -in @('generalize', 'all')) {
    foreach ($Spec in @(
        @{ Context='128'; Order=@('parent','candidate') },
        @{ Context='4096'; Order=@('candidate','parent') }
    )) {
        foreach ($Arm in $Spec.Order) {
            Invoke-Step "Phase30E ctx$($Spec.Context) $Arm" @(
                $Measure, '--arm', $Arm, '--tag', "CTX$($Spec.Context)",
                '--context', $Spec.Context, '--blocks', '8', '--warmup', '4'
            )
        }
    }
    Invoke-Step 'Phase30E generalization adjudication' @($Generalize)
}
