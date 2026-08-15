param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$ProDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $ProDir
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Nemotron Python environment not found: $Python"
}
if (-not $env:LS_MODEL_DIR) {
    $env:LS_MODEL_DIR = 'nemotron_3_5_lightning_v35'
}

Set-Location -LiteralPath $Repo
$Log = Join-Path $ProDir ("v3_{0}_console.log" -f $Mode)

function Invoke-V3Python([string]$Script, [string[]]$Arguments) {
    Write-Host ""
    Write-Host ("=== {0} ===" -f (Split-Path -Leaf $Script)) -ForegroundColor Cyan
    # Native stderr warnings from CuPy are not PowerShell failures; use the
    # process exit code as the authority while still showing stderr.
    $old = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python $Script @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
        $rc = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $old
    }
    if ($rc -ne 0) {
        throw "V3 script failed with exit code $rc. See $Log"
    }
}

"PRO V3 run $(Get-Date -Format o) mode=$Mode" | Set-Content -LiteralPath $Log
"Git HEAD: $(git rev-parse HEAD)" | Add-Content -LiteralPath $Log
"Model: $env:LS_MODEL_DIR" | Add-Content -LiteralPath $Log

Invoke-V3Python (Join-Path $ProDir 'graph_safe_v3.py') @('--mode', $Mode)
Invoke-V3Python (Join-Path $ProDir 'selective_ervf_v3.py') @('--mode', $Mode)

Write-Host ""
Write-Host "V3 finished." -ForegroundColor Green
Write-Host "Send/push these files:"
Write-Host "  pro_research/results/PRO_V3_G0S_GRAPH_SAFE.json"
Write-Host "  pro_research/results/PRO_V3_G1B_SELECTIVE_ERVF.json"
Write-Host "  pro_research/v3_${Mode}_console.log"
