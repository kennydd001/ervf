$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4') {
    throw "Refusing C1 on branch '$Branch'; expected pro-s100-nativefp4"
}

$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing Nemotron venv: $Py" }

Write-Host '=== S100 native NVFP4 C1 repack audit ==='
git rev-parse HEAD
& $Py 'pro_research\diag_native_nvfp4_c1_repack_v2.py'
if ($LASTEXITCODE -ne 0) {
    Write-Host "C1 returned $LASTEXITCODE. Result JSON is preserved; do not retune mapping/gates post-hoc." -ForegroundColor Yellow
}

$Result = 'pro_research\results\native_nvfp4\C1_REPACK_AUDIT.json'
if (-not (Test-Path $Result)) { throw "No C1 result: $Result" }
Get-Content $Result | Select-String '"status"|"padding_fraction_of_natural"|"C1_G|"C1_P'

Write-Host ''
Write-Host 'To commit the immutable result after inspection:'
Write-Host '  .\pro_research\PUSH_S100_NATIVE_NVFP4_RESULTS.ps1'
