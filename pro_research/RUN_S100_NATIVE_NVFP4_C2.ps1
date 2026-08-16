$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2') { throw "Expected pro-s100-nativefp4-c2, got '$Branch'" }
$Py = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing venv: $Py" }
$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "=== S100 native NVFP4 C2 Torch contract ==="
Write-Host "GPU memory.used = $used MiB"
git status --short
git rev-parse HEAD
if ([int]$used -gt 1024) { throw "GPU busy ($used MiB); refusing timing" }
& $Py 'pro_research\diag_native_nvfp4_c2_torch_contract.py'
$rc = $LASTEXITCODE
$Result = Join-Path $Repo 'pro_research\results\native_nvfp4\C2_TORCH_CONTRACT.json'
if (-not (Test-Path $Result)) { throw "C2 produced no result JSON (rc=$rc)" }
Write-Host '=== Result ==='
Get-Content $Result | Select-String '"status"|"G4_M1_known_value_executes"|"P1_M1_QLIKE_lt_0_20ms"|"P3_M2_vs_M1_QLIKE_time_ratio_le_1_40"|"C3_real_checkpoint_open"|"error"'
if ($rc -ne 0) { throw "C2 technical failure, rc=$rc" }
