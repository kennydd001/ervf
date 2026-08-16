$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') {
    throw "Expected branch 'pro-s100-nativefp4-c2b', got '$Branch'"
}

Write-Host '=== S100 native NVFP4 C2b / isolated Torch 2.12.1+cu132 ==='
git status --short
git rev-parse HEAD

$BootstrapPy = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $BootstrapPy)) {
    throw "Known Python bootstrap interpreter missing: $BootstrapPy"
}

$Venv = Join-Path $Repo '.venv-fp4-c2b'
$Py = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $Py)) {
    Write-Host 'Creating isolated .venv-fp4-c2b (the Nemotron venv is not modified)...'
    & $BootstrapPy -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv-fp4-c2b' }
}

# Pin the exact software contract. Query first so reruns do not reinstall a
# multi-GB wheel when the environment is already correct.
$TorchProbe = & $Py -c "import torch; print(torch.__version__); print(torch.version.cuda)" 2>$null
$NeedTorch = $true
if ($LASTEXITCODE -eq 0 -and $TorchProbe.Count -ge 2) {
    $tv = [string]$TorchProbe[0]
    $cv = [string]$TorchProbe[1]
    if ($tv.StartsWith('2.12.1') -and $cv.StartsWith('13.2')) { $NeedTorch = $false }
}
if ($NeedTorch) {
    Write-Host 'Installing official PyTorch 2.12.1 CUDA 13.2 wheel into isolated venv...'
    & $Py -m pip install --disable-pip-version-check --no-cache-dir `
        torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch 2.12.1+cu132 installation failed' }
}

Write-Host 'Isolated environment:'
& $Py -c "import sys,torch; print(sys.executable); print('torch='+torch.__version__); print('torch.cuda='+str(torch.version.cuda)); print('cuda.available='+str(torch.cuda.is_available())); print('device='+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'device=None'); print('cap='+str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else 'cap=None')"
if ($LASTEXITCODE -ne 0) { throw 'Isolated Torch environment probe failed' }

$used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
Write-Host "GPU memory.used = $used MiB"
if ([int]$used -gt 1024) { throw "GPU already busy ($used MiB); refusing scientific timing" }

$env:PYTHONHASHSEED = '0'
& $Py 'pro_research\diag_native_nvfp4_c2b_torch212.py'
$rc = $LASTEXITCODE
$Result = Join-Path $Repo 'pro_research\results\native_nvfp4\C2B_TORCH212_CONTRACT.json'
if (-not (Test-Path $Result)) { throw "No C2b result JSON produced (runner rc=$rc)" }
if ($rc -ne 0) {
    Write-Host "C2b runner returned $rc; preserved result follows." -ForegroundColor Yellow
    Get-Content $Result | Select-String '"status"|"message"|"error"'
    exit $rc
}

& $Py 'pro_research\verify_s100_native_nvfp4_c2b.py'
if ($LASTEXITCODE -ne 0) { throw 'Independent C2b verifier failed' }

Write-Host '=== C2b result ==='
Get-Content $Result | Select-String '"status"|"torch_version"|"torch_cuda_version"|"M1_QLIKE_p50_ms"|"M2_QLIKE_p50_ms"|"M1_MAMBA_IN_p50_ms"|"M2_MAMBA_IN_p50_ms"|"qlike_M2_over_M1"|"mamba_in_M2_over_M1"|"C3_real_checkpoint_open"'
