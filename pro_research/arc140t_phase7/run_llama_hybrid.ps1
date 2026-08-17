param(
    [Parameter(Mandatory=$true)][string]$BinDir,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$OutDir
)
$ErrorActionPreference='Continue'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Cli=Join-Path $BinDir 'llama-cli.exe'
$Bench=Join-Path $BinDir 'llama-bench.exe'
if(-not(Test-Path $Bench)){throw "Missing $Bench"}
if(-not(Test-Path $Cli)){throw "Missing $Cli"}
if(-not(Test-Path $Model)){throw "Missing $Model"}

$Devices=& $Cli --list-devices 2>&1
$Devices | Out-File (Join-Path $OutDir 'devices.txt') -Encoding utf8
$Help=& $Bench --help 2>&1
$Help | Out-File (Join-Path $OutDir 'bench-help.txt') -Encoding utf8
$Version=& $Cli --version 2>&1
$Version | Out-File (Join-Path $OutDir 'llama-version.txt') -Encoding utf8
(Get-FileHash $Model -Algorithm SHA256).Hash | Set-Content (Join-Path $OutDir 'model.sha256')

function Get-Device($kind,$needle){
    foreach($Line in $Devices){
        $S=[string]$Line
        if($S -match "(?i)^\s*([A-Za-z]+[0-9]+)\s*:" -and $matches[1] -like "$kind*" -and $S -match $needle){
            return $matches[1]
        }
    }
    return $null
}
$Rtx=Get-Device 'CUDA' 'NVIDIA|RTX'
$Arc=Get-Device 'Vulkan' 'Intel.*Arc|Arc.*140T|Intel'
$ArcSycl=Get-Device 'SYCL' 'Intel.*Arc|Intel'
$ArcOv=Get-Device 'OPENVINO' 'Intel|GPU'
[ordered]@{rtx=$Rtx;arc_vulkan=$Arc;arc_sycl=$ArcSycl;arc_openvino=$ArcOv;raw=@($Devices)} |
    ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutDir 'device-map.json') -Encoding utf8

$script:Ledger=@()
function RunBench([string]$Name,[string[]]$Args,[hashtable]$Env=@{}){
    $Json=Join-Path $OutDir ($Name+'.json')
    $Err=Join-Path $OutDir ($Name+'.stderr.txt')
    $Cmd="$Bench "+($Args -join ' ')
    foreach($K in $Env.Keys){Set-Item "Env:$K" $Env[$K]}
    try{
        $Raw=& $Bench @Args -o json -oe none 2> $Err
        $Code=$LASTEXITCODE
        if($Code -eq 0){
            $Raw | Out-File $Json -Encoding utf8
            $script:Ledger += [pscustomobject]@{name=$Name;status='measured';exit=$Code;command=$Cmd;json=$Json}
        } else {
            $Raw | Out-File $Json -Encoding utf8
            $script:Ledger += [pscustomobject]@{name=$Name;status='failed';exit=$Code;command=$Cmd;json=$Json}
        }
    } catch {
        $_ | Out-File $Err -Append
        $script:Ledger += [pscustomobject]@{name=$Name;status='technical_failure';exit=-1;command=$Cmd;json=$Json}
    } finally {
        foreach($K in $Env.Keys){Remove-Item "Env:$K" -ErrorAction SilentlyContinue}
    }
}

$Common=@('-m',$Model,'-r','5','-p','512','-n','128','--delay','1')
RunBench 'cpu-only' ($Common+@('-dev','none','-ngl','0'))
if($Rtx){
    RunBench 'rtx-only-all' ($Common+@('-dev',$Rtx,'-sm','none','-ngl','all'))
    RunBench 'rtx-only-auto' ($Common+@('-dev',$Rtx,'-sm','none','-ngl','auto'))
}
if($Arc){RunBench 'arc-vulkan-only' ($Common+@('-dev',$Arc,'-sm','none','-ngl','all'))}
if($ArcSycl){RunBench 'arc-sycl-only' ($Common+@('-dev',$ArcSycl,'-sm','none','-ngl','all'))}
if($ArcOv){RunBench 'arc-openvino-only' ($Common+@('-dev',$ArcOv,'-sm','none','-ngl','all'))}

$HybridConfigs=@{}
if($Rtx -and $Arc){
    $HybridConfigs['hybrid-auto']=@('-dev',"$Rtx,$Arc",'-sm','layer','-ngl','all','-fitt','1024,512')
    RunBench 'hybrid-auto' ($Common+$HybridConfigs['hybrid-auto'])
    foreach($Headroom in @('512,256','1024,512','1536,512','2048,1024')){
        $Safe=$Headroom.Replace(',','-')
        $Name="hybrid-fit-$Safe"
        $Args=@('-dev',"$Rtx,$Arc",'-sm','layer','-ngl','all','-fitt',$Headroom)
        $HybridConfigs[$Name]=$Args
        RunBench $Name ($Common+$Args)
    }
    $Ratios=5..95 | Where-Object {$_%5 -eq 0}
    foreach($Pct in $Ratios){
        $ArcPct=100-$Pct
        $Name="hybrid-fwd-$Pct-$ArcPct"
        $Args=@('-dev',"$Rtx,$Arc",'-sm','layer','-ngl','all','-ts',"$Pct,$ArcPct")
        $HybridConfigs[$Name]=$Args
        RunBench $Name ($Common+$Args)
    }
    foreach($Pct in $Ratios){
        $RtxPct=100-$Pct
        $Name="hybrid-rev-$Pct-$RtxPct"
        $Args=@('-dev',"$Arc,$Rtx",'-sm','layer','-ngl','all','-ts',"$Pct,$RtxPct")
        $HybridConfigs[$Name]=$Args
        RunBench $Name ($Common+$Args)
    }
    RunBench 'hybrid-auto-cuda-queues-4x' ($Common+$HybridConfigs['hybrid-auto']) @{'CUDA_SCALE_LAUNCH_QUEUES'='4x'}

    # Pick fastest TG configuration from measured hybrid JSONs.
    $BestName=$null;$BestTG=-1.0
    foreach($Name in $HybridConfigs.Keys){
        $P=Join-Path $OutDir ($Name+'.json')
        if(-not(Test-Path $P)){continue}
        try{
            $Rows=Get-Content $P -Raw|ConvertFrom-Json
            foreach($Row in @($Rows)){
                if([int]$Row.n_gen -gt 0 -and [int]$Row.n_prompt -eq 0 -and [double]$Row.avg_ts -gt $BestTG){
                    $BestTG=[double]$Row.avg_ts;$BestName=$Name
                }
            }
        }catch{}
    }
    if($BestName){
        $BestArgs=$HybridConfigs[$BestName]
        "best=$BestName tg=$BestTG args=$($BestArgs -join ' ')" | Set-Content (Join-Path $OutDir 'best-hybrid.txt')
        foreach($KV in @('f16','q8_0','q4_0')){
            RunBench "best-kv-$KV" ($Common+$BestArgs+@('-ctk',$KV,'-ctv',$KV))
        }
        foreach($Depth in @(4096,16384,32768)){
            RunBench "best-depth-$Depth" (@('-m',$Model,'-r','5','-p','0','-n','128','-d',"$Depth",'--delay','1')+$BestArgs)
        }
        foreach($FA in @('auto','on','off')){
            RunBench "best-fa-$FA" ($Common+$BestArgs+@('-fa',$FA))
        }
        RunBench 'best-op-offload-default' ($Common+$BestArgs)
        if(($Help -join "`n") -match 'no-op-offload'){
            RunBench 'best-no-op-offload' ($Common+$BestArgs+@('--no-op-offload'))
        }
        # Experimental tensor-parallel compatibility test. Failure is data.
        RunBench 'tensor-experimental' ($Common+@('-dev',"$Rtx,$Arc",'-sm','tensor','-ngl','all','-ctk','f16','-ctv','f16','-fa','on'))
    }
}
$script:Ledger | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutDir 'ledger.json') -Encoding utf8
