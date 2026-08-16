param(
    [ValidateSet('smoke','full')]
    [string]$Mode = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Runner = Join-Path $Repo 'pro_research\v13_qkv_steady.py'
$OutDir = Join-Path $Repo 'pro_research\results\v12_async'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $OutDir ("V13_QKV_${Mode}_${Stamp}.console.log")
& $Python $Runner --mode $Mode 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "V13 QKV returned code $LASTEXITCODE" }
