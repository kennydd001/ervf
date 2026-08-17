param([Parameter(Mandatory=$true)][string]$OutDir)
$ErrorActionPreference='Continue'
$Src=Join-Path $PSScriptRoot 'd3d12_cross_adapter_probe.cpp'
$Build=Join-Path $OutDir 'd3d12-build';New-Item -ItemType Directory -Force $Build|Out-Null
$CMake=Get-Command cmake -ErrorAction SilentlyContinue
if(-not$CMake){'{"status":"cmake_missing"}'|Set-Content (Join-Path $OutDir 'd3d12_cross_adapter.json');return}
$CMakeLists=@"
cmake_minimum_required(VERSION 3.20)
project(s100_d3d12_probe LANGUAGES CXX)
add_executable(s100_d3d12_probe "$($Src.Replace('\','/'))")
target_compile_features(s100_d3d12_probe PRIVATE cxx_std_17)
target_link_libraries(s100_d3d12_probe PRIVATE d3d12 dxgi dxguid)
"@
$CMakeLists|Set-Content (Join-Path $Build 'CMakeLists.txt') -Encoding utf8
cmake -S $Build -B (Join-Path $Build 'out') -A x64 *> (Join-Path $Build 'configure.log')
if($LASTEXITCODE-ne0){'{"status":"cmake_configure_failed"}'|Set-Content (Join-Path $OutDir 'd3d12_cross_adapter.json');return}
cmake --build (Join-Path $Build 'out') --config Release *> (Join-Path $Build 'build.log')
$Exe=Join-Path $Build 'out\Release\s100_d3d12_probe.exe'
if(-not(Test-Path $Exe)){'{"status":"build_failed"}'|Set-Content (Join-Path $OutDir 'd3d12_cross_adapter.json');return}
& $Exe | Tee-Object -FilePath (Join-Path $OutDir 'd3d12_cross_adapter.json')
