$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4') { throw "Refusing C0 on branch '$Branch'; expected pro-s100-nativefp4" }
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\diag_native_nvfp4_capabilities.py'
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }

Write-Host "Repository : $Repo"
Write-Host "Branch     : $Branch"
Write-Host 'Experiment : S100 native NVFP4 C0 capability/format audit'

& $Python $Runner
$code = $LASTEXITCODE
if ($code -ne 0) { throw "C0 runner returned native exit code ${code}" }

Write-Host 'C0 completed. A gate_failed result is scientific evidence, not a technical failure.' -ForegroundColor Green
