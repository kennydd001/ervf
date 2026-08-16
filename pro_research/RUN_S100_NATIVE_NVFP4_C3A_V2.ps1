$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') { throw "Expected branch pro-s100-nativefp4-c2b, got $Branch" }

Write-Host '=== S100 native NVFP4 C3A-v2 / corrected TorchAO scale layout ==='
git status --short
git rev-parse HEAD

$Py = Join-Path $Repo '.venv-fp4-c2b\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Missing isolated environment: $Py" }
$Index = Join-Path $Repo 'models\nemotron_3_5_lightning_v35\model.safetensors.index.json'
if (-not (Test-Path $Index)) { throw "Missing checkpoint index: $Index" }

& $Py -c "import torch,torch.nn.functional as F; print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda)); print('gpu='+torch.cuda.get_device_name(0)); print('cap='+str(torch.cuda.get_device_capability(0))); print('BlockWise1x16='+str(F.ScalingType.BlockWise1x16)); print('swizzle='+str(F.SwizzleType.SWIZZLE_32_4_4))"
if ($LASTEXITCODE -ne 0) { throw 'Environment probe failed' }

$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

$env:PYTHONHASHSEED = '0'
& $Py 'pro_research\diag_native_nvfp4_c3a_real_weight_v2.py'
if ($LASTEXITCODE -ne 0) { throw 'C3A-v2 diagnostic/preflight failed; preserve both JSON results' }

& $Py 'pro_research\verify_s100_native_nvfp4_c3a_layout_v2.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent C3A-v2 layout verifier failed' }

& $Py 'pro_research\verify_s100_native_nvfp4_c3a_real_weight.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent C3A real-checkpoint verifier failed' }

Write-Host '=== C3A-v2 PASS ===' -ForegroundColor Green
$Compact = @'
import json, pathlib
pre=json.loads(pathlib.Path(r"pro_research/results/native_nvfp4/C3A_V2_LAYOUT_PREFLIGHT.json").read_text())
res=json.loads(pathlib.Path(r"pro_research/results/native_nvfp4/C3A_REAL_WEIGHT.json").read_text())
print("v2_preflight_status=", pre.get("status"))
print("layout_witness=", json.dumps(pre.get("layout_witness"), indent=2))
print("nonuniform_native_smoke=", json.dumps(pre.get("nonuniform_native_smoke"), indent=2))
print("c3a_status=", res.get("status"))
print("summary=", json.dumps(res.get("summary"), indent=2))
print("gates=", json.dumps(res.get("gates"), indent=2))
'@
& $Py -c $Compact

Write-Host ''
Write-Host 'Send back:'
Write-Host '  pro_research\results\native_nvfp4\C3A_V2_LAYOUT_PREFLIGHT.json'
Write-Host '  pro_research\results\native_nvfp4\C3A_REAL_WEIGHT.json'
Write-Host '  complete console output + git rev-parse HEAD + git status --short'
