$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') { throw "Expected branch pro-s100-nativefp4-c2b, got $Branch" }

Write-Host '=== S100 native NVFP4 C3B / real causal activations ==='
git status --short
git rev-parse HEAD

$PyFP4 = Join-Path $Repo '.venv-fp4-c2b\Scripts\python.exe'
if (-not (Test-Path $PyFP4)) { throw "Missing FP4 environment: $PyFP4" }

# Runtime experiments predate the isolated FP4 venv, and local checkout names
# differ. Select the first Python that can actually import the V18 dependencies.
$Candidates = @(
    (Join-Path $Repo '.venv-nemotron\Scripts\python.exe'),
    (Join-Path $Repo '.venv\Scripts\python.exe')
)
try { $Candidates += (Get-Command python -ErrorAction Stop).Source } catch {}
$PyRT = $null
foreach ($cand in $Candidates | Select-Object -Unique) {
    if (-not (Test-Path $cand)) { continue }
    & $cand -c "import cupy,numpy; print('runtime-python-ok')" *> $null
    if ($LASTEXITCODE -eq 0) { $PyRT = $cand; break }
}
if (-not $PyRT) { throw 'No Python environment can import cupy + numpy for the V18/V6 runtime capture.' }
Write-Host "Runtime Python = $PyRT"
$Index = Join-Path $Repo 'models\nemotron_3_5_lightning_v35\model.safetensors.index.json'
if (-not (Test-Path $Index)) { throw "Missing real Lightning checkpoint: $Index" }

# C3B is based on the corrected C3A-v2 implementation. A descendant commit is
# fine, but an older checkout is not.
git merge-base --is-ancestor e316da090240ada357e39202522c9acfcf89abe9 HEAD
if ($LASTEXITCODE -ne 0) { throw 'Checkout is older than required C3A-v2 base e316da0. Run git fetch/pull first.' }

$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

$C3A = 'pro_research\results\native_nvfp4\C3A_REAL_WEIGHT.json'
$C3APre = 'pro_research\results\native_nvfp4\C3A_V2_LAYOUT_PREFLIGHT.json'
if (-not (Test-Path $C3A) -or -not (Test-Path $C3APre)) {
    Write-Host 'C3A-v2 result missing; running frozen parent first...'
    & (Join-Path $Repo 'pro_research\RUN_S100_NATIVE_NVFP4_C3A_V2.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'C3A-v2 parent failed' }
} else {
    & $PyFP4 'pro_research\verify_s100_native_nvfp4_c3a_layout_v2.py'
    if ($LASTEXITCODE -ne 0) { throw 'C3A-v2 layout parent verifier failed' }
    & $PyFP4 'pro_research\verify_s100_native_nvfp4_c3a_real_weight.py'
    if ($LASTEXITCODE -ne 0) { throw 'C3A real-weight parent verifier failed' }
}

# Capture is a separate process/venv so no Torch/CuPy CUDA contexts coexist.
$env:PYTHONHASHSEED = '0'
& $PyRT 'pro_research\capture_native_nvfp4_c3b_realact.py' --tokens 64
if ($LASTEXITCODE -ne 0) { throw 'C3B real-activation capture failed' }

$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU memory.used after capture process = $used MiB"
if ([int]$used -gt 1024) { throw "GPU did not return to idle after capture ($used MiB)" }

& $PyFP4 'pro_research\diag_native_nvfp4_c3b_realact.py'
if ($LASTEXITCODE -ne 0) { throw 'C3B diagnostic technical failure' }

& $PyFP4 'pro_research\verify_s100_native_nvfp4_c3b_realact.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent C3B consistency verifier failed' }

Write-Host '=== C3B completed ===' -ForegroundColor Green
$Compact = @'
import json, pathlib
p=pathlib.Path(r"pro_research/results/native_nvfp4/C3B_REAL_ACTIVATION.json")
d=json.loads(p.read_text())
print("status=", d.get("status"))
print("summary=", json.dumps(d.get("summary"), indent=2))
print("gates=", json.dumps(d.get("gates"), indent=2))
for f in d.get("families", []):
    print("\n",f["label"])
    print(" dynamic quality=", f["quality"]["dynamic"]["metrics"])
    print(" static  quality=", f["quality"]["static_1p10"]["metrics"])
    print(" dynamic M8/M1=", f["prequantized_native_timing"]["dynamic"].get("M8_over_M1"))
    print(" static  M8/M1=", f["prequantized_native_timing"]["static_1p10"].get("M8_over_M1"))
'@
& $PyFP4 -c $Compact

Write-Host ''
Write-Host 'Send back:'
Write-Host '  pro_research\results\native_nvfp4\C3A_V2_LAYOUT_PREFLIGHT.json'
Write-Host '  pro_research\results\native_nvfp4\C3A_REAL_WEIGHT.json'
Write-Host '  pro_research\results\native_nvfp4\C3B_REAL_ACTIVATION.json'
Write-Host '  complete console output + git rev-parse HEAD + git status --short'

