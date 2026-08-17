param([string]$OutDir)
$ErrorActionPreference='Continue'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Obj=[ordered]@{
    created=(Get-Date).ToString('o')
    os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber)
    cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed)
    computer=(Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory)
    memory=@(Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed)
    video=@(Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion,AdapterRAM,VideoProcessor)
}
$Obj | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutDir 'inventory.json') -Encoding utf8
try { nvidia-smi -q | Out-File (Join-Path $OutDir 'nvidia-smi-q.txt') -Encoding utf8 } catch {}
try { nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total,memory.used,power.limit,clocks.mem,clocks.sm --format=csv | Out-File (Join-Path $OutDir 'nvidia-smi.csv') -Encoding utf8 } catch {}
try { vulkaninfo --summary | Out-File (Join-Path $OutDir 'vulkan-summary.txt') -Encoding utf8 } catch {}
try { sycl-ls | Out-File (Join-Path $OutDir 'sycl-ls.txt') -Encoding utf8 } catch {}
try { clinfo | Out-File (Join-Path $OutDir 'clinfo.txt') -Encoding utf8 } catch {}
try { dxdiag /t (Join-Path $OutDir 'dxdiag.txt') | Out-Null } catch {}
try { Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop | Export-Counter -Path (Join-Path $OutDir 'gpu-counters.blg') -FileFormat BLG -Force } catch {}
