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

$Common = Join-Path $ProDir 'common.py'
if (-not (Test-Path -LiteralPath $Common)) {
    & $Python (Join-Path $ProDir 'bootstrap.py')
    if ($LASTEXITCODE -ne 0) { throw "PRO payload expansion failed with code $LASTEXITCODE" }
    # The verified payload replaces this bootstrap wrapper with the full runner.
    & (Join-Path $ProDir 'INSTALL_AND_RUN.ps1') -Mode $Mode
    exit $LASTEXITCODE
}

# Safety fallback if source files already exist but this small wrapper remained.
& $Python (Join-Path $ProDir 'bootstrap.py')
if ($LASTEXITCODE -ne 0) { throw "PRO payload refresh failed with code $LASTEXITCODE" }
& (Join-Path $ProDir 'INSTALL_AND_RUN.ps1') -Mode $Mode
exit $LASTEXITCODE
