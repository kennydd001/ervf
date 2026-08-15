param(
    [ValidateSet('install','smoke','full','graph','dense','epoch','verify','report')]
    [string]$Mode = 'install'
)

$ErrorActionPreference = 'Stop'
$ProDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $ProDir
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Nemotron Python environment not found: $Python"
}

if (-not $env:LS_MODEL_DIR) {
    $env:LS_MODEL_DIR = 'nemotron_3_5_lightning_v35'
}

$modelCandidate = $env:LS_MODEL_DIR
if (-not [System.IO.Path]::IsPathRooted($modelCandidate)) {
    $modelCandidate = Join-Path (Join-Path $Repo 'models') $modelCandidate
}
if (-not (Test-Path -LiteralPath $modelCandidate)) {
    throw "Model directory not found: $modelCandidate`nSet LS_MODEL_DIR to the correct local model directory."
}

Set-Location -LiteralPath $Repo
New-Item -ItemType Directory -Force -Path (Join-Path $ProDir 'results') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProDir 'results\history') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProDir 'results\logs') | Out-Null

Write-Host "Repository : $Repo"
Write-Host "Python     : $Python"
Write-Host "Model      : $modelCandidate"
Write-Host "Mode       : $Mode"

function Invoke-ProPython([string[]]$Arguments) {
    # Windows PowerShell 5.1 promotes native stderr lines to ErrorRecords.
    # CuPy can emit a CUDA_PATH UserWarning on stderr while the process itself
    # remains usable. Do not let ErrorActionPreference=Stop create a false
    # failure; the native process exit code remains authoritative.
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments
        $nativeRc = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($nativeRc -ne 0) {
        throw "PRO Python command exited with code $nativeRc. Read pro_research\results\logs."
    }
}

switch ($Mode) {
    'install' {
        Invoke-ProPython @('-m', 'compileall', '-q', $ProDir)
        Invoke-ProPython @((Join-Path $ProDir 'ervf_dense.py'), '--selftest')
        Write-Host 'PRO pack installed and CPU selftest passed.' -ForegroundColor Green
        Write-Host 'Next: .\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke'
    }
    'smoke'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'smoke') }
    'full'   { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'full') }
    'graph'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'graph') }
    'dense'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'dense') }
    'epoch'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'epoch') }
    'verify' { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'verify') }
    'report' { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'report') }
}
