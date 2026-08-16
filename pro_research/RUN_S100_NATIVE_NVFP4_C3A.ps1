$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') {
    throw "Expected branch 'pro-s100-nativefp4-c2b', got '$Branch'"
}

Write-Host '=== S100 native NVFP4 C3A / REAL CHECKPOINT WEIGHTS ==='
Write-Host 'This test is correctness-first. It does not modify the checkpoint.'
git status --short
git rev-parse HEAD

$Py = Join-Path $Repo '.venv-fp4-c2b\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    throw "Missing isolated C2b environment: $Py. Run pro_research\RUN_S100_NATIVE_NVFP4_C2B.ps1 once first."
}

$ModelIndex = Join-Path $Repo 'models\nemotron_3_5_lightning_v35\model.safetensors.index.json'
if (-not (Test-Path $ModelIndex)) {
    throw "Missing Lightning checkpoint index: $ModelIndex"
}

Write-Host 'Environment:'
& $Py -c "import sys,torch,torch.nn.functional as F; print(sys.executable); print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda)); print('gpu='+torch.cuda.get_device_name(0)); print('cap='+str(torch.cuda.get_device_capability(0))); print('scaled_mm='+str(hasattr(F,'scaled_mm'))); print('BlockWise1x16='+str(F.ScalingType.BlockWise1x16)); print('TensorWise='+str(F.ScalingType.TensorWise)); print('swizzle='+str(F.SwizzleType.SWIZZLE_32_4_4))"
if ($LASTEXITCODE -ne 0) { throw 'C3A environment probe failed' }

$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

$env:PYTHONHASHSEED = '0'
& $Py 'pro_research\diag_native_nvfp4_c3a_real_weight.py'
$rc = $LASTEXITCODE
$Result = Join-Path $Repo 'pro_research\results\native_nvfp4\C3A_REAL_WEIGHT.json'
if (-not (Test-Path $Result)) { throw "No C3A result JSON produced (runner rc=$rc)" }
if ($rc -ne 0) {
    Write-Host "C3A diagnostic returned $rc; preserving result." -ForegroundColor Yellow
    Get-Content $Result | Select-String '"status"|"error"|"message"'
    exit $rc
}

& $Py 'pro_research\verify_s100_native_nvfp4_c3a_real_weight.py'
$verifyRc = $LASTEXITCODE
if ($verifyRc -ne 0) {
    Write-Host 'Independent C3A verifier FAILED. Do not interpret performance numbers as a breakthrough.' -ForegroundColor Red
    exit $verifyRc
}

Write-Host '=== C3A compact result ==='
$Compact = @'
import json, pathlib
p = pathlib.Path(r"pro_research/results/native_nvfp4/C3A_REAL_WEIGHT.json")
d = json.loads(p.read_text())
print("status=", d.get("status"))
print("summary=", json.dumps(d.get("summary"), indent=2))
print("gates=", json.dumps(d.get("gates"), indent=2))
for f in d.get("families", []):
    print(f["label"], "nrmse=", f["native"]["M1"]["reference_metrics_first_row"]["normalized_rmse"],
          "cos=", f["native"]["M1"]["reference_metrics_first_row"]["cosine"],
          "M8/M1=", f["cold_timing"].get("M8_over_M1"),
          "L2x=", f["cold_timing"].get("working_set_over_l2"))
'@
& $Py -c $Compact

Write-Host ''
Write-Host 'Send back:'
Write-Host '  1) the complete console output'
Write-Host '  2) pro_research\results\native_nvfp4\C3A_REAL_WEIGHT.json'
Write-Host '  3) git rev-parse HEAD'
Write-Host '  4) git status --short'
