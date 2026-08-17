param([string]$OutDir)
$ErrorActionPreference='Continue'
$Result=[ordered]@{status='started';source=$null;bin=$null;commit=$null;build_log=$null}
function FindBin{
    $Candidates=@()
    foreach($Name in @('llama-cli.exe','llama-bench.exe')){
        try{$C=Get-Command $Name -ErrorAction Stop;$Candidates+=$C.Source}catch{}
    }
    foreach($Root in @(
        "$env:LOCALAPPDATA\S100ArcLab\llama.cpp\build-s100\bin\Release",
        "$env:LOCALAPPDATA\S100ArcLab\llama.cpp\build-s100\bin",
        "$env:USERPROFILE\Documents\ChatGPT\New project\third_party\llama.cpp\build\bin\Release",
        "$env:USERPROFILE\Documents\ChatGPT\New project\third_party\llama.cpp\build\bin"
    )){
        if((Test-Path (Join-Path $Root 'llama-cli.exe')) -and (Test-Path (Join-Path $Root 'llama-bench.exe'))){return $Root}
    }
    if($Candidates.Count){
        $D=Split-Path $Candidates[0]
        if(Test-Path (Join-Path $D 'llama-bench.exe')){return $D}
    }
    return $null
}
$Found=FindBin
if($Found){$Result.status='existing';$Result.bin=$Found;$Result|ConvertTo-Json|Set-Content (Join-Path $OutDir 'llama-bootstrap.json');exit 0}

$Git=Get-Command git -ErrorAction SilentlyContinue;$Cmake=Get-Command cmake -ErrorAction SilentlyContinue
if(-not$Git -or -not$Cmake){$Result.status='missing_build_tools';$Result|ConvertTo-Json|Set-Content (Join-Path $OutDir 'llama-bootstrap.json');exit 0}
$Src="$env:LOCALAPPDATA\S100ArcLab\llama.cpp";$Build=Join-Path $Src 'build-s100';New-Item -ItemType Directory -Force (Split-Path $Src) | Out-Null
if(-not(Test-Path (Join-Path $Src '.git'))){git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $Src}
else{git -C $Src fetch origin;git -C $Src switch master;git -C $Src pull --ff-only}
$Result.source=$Src;$Result.commit=(git -C $Src rev-parse HEAD).Trim()
$Log=Join-Path $OutDir 'llama-build.txt';$Result.build_log=$Log
cmake -S $Src -B $Build -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF *>&1 | Tee-Object $Log
if($LASTEXITCODE -eq 0){cmake --build $Build --config Release --target llama-cli llama-bench -j *>&1 | Tee-Object $Log -Append}
$Bin=Join-Path $Build 'bin\Release';if(-not(Test-Path (Join-Path $Bin 'llama-cli.exe'))){$Bin=Join-Path $Build 'bin'}
if((Test-Path (Join-Path $Bin 'llama-cli.exe')) -and (Test-Path (Join-Path $Bin 'llama-bench.exe'))){$Result.status='built';$Result.bin=$Bin}else{$Result.status='build_failed'}
$Result|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutDir 'llama-bootstrap.json') -Encoding utf8
