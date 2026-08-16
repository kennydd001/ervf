param(
    [ValidateSet('install','smoke','full','architecture','overnight','verify','report')]
    [string]$Mode = 'smoke'
)

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent (Split-Path -Parent $Here)
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'
$Model = Join-Path $Repo 'models\nemotron_3_5_lightning_v35'
$Results = Join-Path $Repo 'pro_research\results\pro_max_v2'

if (-not (Test-Path $Python)) { throw "Python venv ontbreekt: $Python" }
if (-not (Test-Path $Model)) { throw "Model ontbreekt: $Model" }

$branch = (& git -C $Repo branch --show-current).Trim()
if ($branch -ne 'pro-max-v2') {
    Write-Warning "Huidige branch is '$branch'; verwacht 'pro-max-v2'."
}

$env:LS_MODEL_DIR = $Model
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$log = Join-Path $Results "PV2_${Mode}_${stamp}.console.log"

Write-Host "Repository : $Repo"
Write-Host "Branch     : $branch"
Write-Host "Python     : $Python"
Write-Host "Model      : $Model"
Write-Host "Mode       : $Mode"
Write-Host "Log        : $log"

& $Python (Join-Path $Here 'campaign.py') $Mode 2>&1 | Tee-Object -FilePath $log
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Warning "Campagne eindigde met return code $rc. Bekijk de JSON/logs; een technische failure is geen negatieve hypothese."
}
exit $rc
