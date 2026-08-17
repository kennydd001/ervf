param(
    [Parameter(Mandatory=$true)][string]$ArcRepo,
    [Parameter(Mandatory=$true)][string]$S100Repo,
    [Parameter(Mandatory=$true)][string]$ArcPython,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$ShapeFile
)
$ErrorActionPreference='Continue'
$Result = Join-Path $S100Repo 'pro_research\results\S100_PHASE4_FRESH_QFAST_SMOKE_CAND_A.json'
$Py = Join-Path $S100Repo '.venv-nemotron\Scripts\python.exe'
$Arm = Join-Path $S100Repo 'pro_research\s100_phase4_fresh_arm.py'
$Shape = $ShapeFile
$Rows = @()

function Run-QFast([string]$Label) {
    Push-Location $S100Repo
    & $Py $Arm --profile qfast --role cand_a --mode smoke
    Pop-Location
    if (Test-Path $Result) {
        $D=Get-Content $Result -Raw|ConvertFrom-Json
        $script:Rows += [pscustomobject]@{label=$Label;p50_ms=[double]$D.timing.p50;p95_ms=[double]$D.timing.p95;vram_mib=[int]$D.vram_mib;smi_before=$D.smi_before;smi_after=$D.smi_after}
    }
}
Run-QFast 'base_a'
$LoadScript=Join-Path $ArcRepo 'pro_research\arc140t_phase8\arc_load_openvino.py'
$LoadLog=Join-Path $OutDir 'arc_load.log'
$ArgList=@("`"$LoadScript`"",'--shape',"`"$Shape`"",'--seconds','75')
$Proc=Start-Process -FilePath $ArcPython -ArgumentList $ArgList -PassThru -NoNewWindow -RedirectStandardOutput $LoadLog -RedirectStandardError ($LoadLog+'.err')
Start-Sleep -Seconds 5
Run-QFast 'arc_load'
try{$Proc.WaitForExit(60000)}catch{try{$Proc.Kill()}catch{}}
Start-Sleep -Seconds 3
Run-QFast 'base_b'
$Base=($Rows|Where-Object label -like 'base*'|Measure-Object p50_ms -Average).Average
$Load=($Rows|Where-Object label -eq 'arc_load'|Select-Object -First 1).p50_ms
$Obj=[ordered]@{kind='s100_phase8_qfast_arc_interference';rows=$Rows;base_midpoint_ms=$Base;arc_load_ms=$Load;regression_fraction=if($Base){($Load-$Base)/$Base}else{$null}}
$Obj|ConvertTo-Json -Depth 10|Set-Content (Join-Path $OutDir 'qfast_arc_interference.json') -Encoding utf8
$Obj|ConvertTo-Json -Depth 10
