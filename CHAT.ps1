# One command, then chat. Starts the Lightning runtime on the V18 stack
# (the 51 tok/s record path), serves a built-in web UI, and opens the browser.
#
#     .\CHAT.ps1
#
# Nothing to install: the UI is served by the same process. Any
# OpenAI-compatible client (llama.cpp web UI, Open WebUI, LM Studio) can also
# point at http://127.0.0.1:<port>/v1 instead.
param(
    [int]$Port = 8080,
    [ValidateSet('v18','v6')][string]$Stack = 'v18',
    [int]$Capacity = 72,
    [switch]$NoBrowser,
    # -Lan binds past loopback so a phone or another PC on the same network can
    # chat. There is NO authentication: anyone who can reach this machine can
    # use the GPU. Safe on a home LAN, not on an open or shared one.
    [switch]$Lan
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot '.venv-nemotron\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Python venv not found: $Py" }

# A leftover run holds the pinned host buffers and the next start fails with
# cudaErrorAlreadyMapped, which reads like a harness bug and is not one.
$stale = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like '*serve_openai*' }
if ($stale) {
    Write-Host "Stopping $($stale.Count) leftover server process(es)..." -ForegroundColor Yellow
    $stale | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Seconds 4
}

$used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
               Select-Object -First 1).Trim()
if ($used -gt 1024) {
    Write-Warning "GPU already using $used MiB. Another job may be running; " +
                  "H-SCALE needs ~492 MiB free and will fall back to v6."
}

Write-Host ""
Write-Host "  Nemotron 3.5 Lightning - stack $Stack" -ForegroundColor Cyan
Write-Host "  Loading weights, this takes ~30s the first time." -ForegroundColor DarkGray
Write-Host "  Chat UI:  http://127.0.0.1:$Port/" -ForegroundColor Green
Write-Host "  OpenAI:   http://127.0.0.1:$Port/v1   (for llama.cpp UI, Open WebUI, ...)" -ForegroundColor DarkGray
Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($p)
        for ($i = 0; $i -lt 90; $i++) {
            try {
                Invoke-WebRequest "http://127.0.0.1:$p/health" -TimeoutSec 2 -UseBasicParsing | Out-Null
                Start-Process "http://127.0.0.1:$p/"
                break
            } catch { Start-Sleep -Seconds 2 }
        }
    } -ArgumentList $Port | Out-Null
}

$BindHost = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
if ($Lan) {
    Write-Host "  -Lan: binding 0.0.0.0. No authentication - only do this on a network you trust." -ForegroundColor Yellow
    Write-Host ""
}

& $Py 'scripts\lightningstream_nemotron\serve_openai.py' `
    --host $BindHost --port $Port --stack $Stack --capacity $Capacity
