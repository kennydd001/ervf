$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') { throw "Expected branch pro-s100-nativefp4-c2b, got $Branch" }

Write-Host '=== S100 native NVFP4 C3B / REAL V18 ACTIVATIONS ==='
git status --short
git rev-parse HEAD

$PyRuntime = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$PyFP4 = Join-Path $Repo '.venv-fp4-c2b\Scripts\python.exe'
if (-not (Test-Path $PyRuntime)) { throw "Missing runtime environment: $PyRuntime" }
if (-not (Test-Path $PyFP4)) { throw "Missing FP4 environment: $PyFP4" }

$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'serve_openai|bench_|diag_native_nvfp4_c3b|capture_native_nvfp4_c3b|-m pip' }
if ($busy) {
    $busy | Select-Object ProcessId, CommandLine
    throw 'Python workload/install is active. Stop it cleanly before C3B.'
}

Write-Host 'Runtime environment:'
& $PyRuntime -c "import sys,numpy,cupy; print(sys.executable); print('numpy='+numpy.__version__); print('cupy='+cupy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime environment probe failed' }
Write-Host 'Isolated native FP4 environment:'
& $PyFP4 -c "import sys,torch,torch.nn.functional as F; print(sys.executable); print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda)); print('gpu='+torch.cuda.get_device_name(0)); print('cap='+str(torch.cuda.get_device_capability(0))); print('BlockWise1x16='+str(F.ScalingType.BlockWise1x16))"
if ($LASTEXITCODE -ne 0) { throw 'FP4 environment probe failed' }

$used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$util = [int](& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU preflight: memory.used=$used MiB utilization=$util%"
if ($used -gt 1024 -or $util -gt 10) { throw "GPU not idle enough for C3B (memory=$used MiB util=$util%)" }

$env:PYTHONHASHSEED = '0'
Write-Host ''
Write-Host '--- Parent verification: C3A-v2 real-checkpoint evidence ---'
& $PyFP4 'pro_research\verify_s100_native_nvfp4_c3a_layout_v2.py'
if ($LASTEXITCODE -ne 0) { throw 'C3A-v2 layout parent verifier failed' }
& $PyFP4 'pro_research\verify_s100_native_nvfp4_c3a_real_weight.py'
if ($LASTEXITCODE -ne 0) { throw 'C3A real-checkpoint parent verifier failed' }

Write-Host ''
Write-Host '--- Phase 1/3: capture 24 real V18 decode states + W4A32 references ---'
& $PyRuntime 'pro_research\capture_native_nvfp4_c3b_real_activations.py'
if ($LASTEXITCODE -ne 0) { throw 'C3B activation capture failed' }

Write-Host ''
Write-Host '--- Phase 2/3: dynamic NVFP4 A + real checkpoint native SM120 ---'
& $PyFP4 'pro_research\diag_native_nvfp4_c3b_real_activations.py'
if ($LASTEXITCODE -ne 0) { throw 'C3B native real-activation diagnostic failed technically' }

Write-Host ''
Write-Host '--- Phase 3/3: independent manifest/top1/gate verifier ---'
& $PyFP4 'pro_research\verify_s100_native_nvfp4_c3b.py'
$verifyRc = $LASTEXITCODE
if ($verifyRc -ne 0) {
    Write-Host 'C3B verifier FAILED. Preserve results; do not interpret performance as an adoption win.' -ForegroundColor Red
    exit $verifyRc
}

Write-Host ''
Write-Host '=== C3B PASS ===' -ForegroundColor Green
$Compact = @'
import json, pathlib
p=pathlib.Path(r"pro_research/results/native_nvfp4/C3B_W4A4_REAL_ACT.json")
d=json.loads(p.read_text())
print("status=", d.get("status"))
print("summary=", json.dumps(d.get("summary"), indent=2))
print("gates=", json.dumps(d.get("gates"), indent=2))
for name,f in d.get("families",{}).items():
    print(name, "M8 quality=", json.dumps(f["quality_by_M"]["M8"]["aggregate"], indent=2))
    print(name, "perf=", json.dumps(f.get("performance"), indent=2))
'@
& $PyFP4 -c $Compact

Write-Host ''
Write-Host 'Send back:'
Write-Host '  1) complete console output'
Write-Host '  2) pro_research\results\native_nvfp4\C3B_CAPTURE.json'
Write-Host '  3) pro_research\results\native_nvfp4\C3B_W4A4_REAL_ACT.json'
Write-Host '  4) pro_research\results\native_nvfp4\C3B_W4A4_VERIFY.json'
Write-Host '  5) git rev-parse HEAD and git status --short'
Write-Host 'Do NOT push result files before analysis.'
