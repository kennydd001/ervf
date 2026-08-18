param(
    [Parameter(Mandatory=$true)][string]$ArcRepo,
    [Parameter(Mandatory=$true)][string]$S100Repo,
    [Parameter(Mandatory=$true)][string]$ArcPython,
    [Parameter(Mandatory=$true)][string]$BankDir,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$LoadSeconds = 180
)
$ErrorActionPreference = 'Stop'
$S100Py = Join-Path $S100Repo '.venv-nemotron\Scripts\python.exe'
$Fresh = Join-Path $S100Repo 'pro_research\s100_phase4_fresh_arm.py'
$Soak = Join-Path $ArcRepo 'pro_research\arc140t_phase8_overnight\arc_full_bank_soak.py'
$Rows = @()

function Run-QFast([string]$Label) {
    Push-Location $S100Repo
    try {
        & $S100Py $Fresh --profile qfast --role cand_a --mode full | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "QFAST $Label failed: $LASTEXITCODE" }
        $Result = Join-Path $S100Repo 'pro_research\results\S100_PHASE4_FRESH_QFAST_FULL_CAND_A.json'
        $D = Get-Content $Result -Raw | ConvertFrom-Json
        $script:Rows += [pscustomobject]@{
            label = $Label
            p50_ms = [double]$D.timing.p50
            p95_ms = [double]$D.timing.p95
            vram_mib = [int]$D.vram_mib
            smi_before = $D.smi_before
            smi_after = $D.smi_after
        }
    }
    finally { Pop-Location }
}

Run-QFast 'base_a'
$Std = "$Out.soak.stdout.txt"
$Err = "$Out.soak.stderr.txt"
$Proc = Start-Process -FilePath $ArcPython -ArgumentList @(
    "`"$Soak`"", '--bank-dir', "`"$BankDir`"", '--seconds', "$LoadSeconds"
) -PassThru -NoNewWindow -RedirectStandardOutput $Std -RedirectStandardError $Err
Start-Sleep -Seconds 12
Run-QFast 'real_arc_load'
try { $Proc.WaitForExit(($LoadSeconds + 60) * 1000) } catch { try { $Proc.Kill() } catch {} }
Start-Sleep -Seconds 5
Run-QFast 'base_b'

$Base = (($Rows | Where-Object label -like 'base*' | Measure-Object p50_ms -Average).Average)
$Load = ($Rows | Where-Object label -eq 'real_arc_load' | Select-Object -First 1).p50_ms
$Obj = [ordered]@{
    kind = 's100_p8_overnight_real_arc_interference'
    rows = $Rows
    base_midpoint_ms = $Base
    arc_load_ms = $Load
    regression_fraction = if ($Base) { ($Load - $Base) / $Base } else { $null }
    soak_stdout = $Std
    soak_stderr = $Err
}
$Obj | ConvertTo-Json -Depth 12 | Set-Content $Out -Encoding utf8
$Obj | ConvertTo-Json -Depth 12
