$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent (Split-Path -Parent $Here)
$Zip = Join-Path $Here 'PRO_MAX_V2_SOURCE.zip'
$ExpectedZipSha256 = 'a8cd2345299112e4146f7de2c0ced0e97006d04970807f19053329ae6ab23816'
$Python = Join-Path $Repo '.venv-nemotron\Scripts\python.exe'

if (-not (Test-Path $Zip)) {
    throw "Source archive ontbreekt: $Zip"
}

$ActualZipSha256 = (Get-FileHash -Path $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualZipSha256 -ne $ExpectedZipSha256) {
    throw "Archive SHA-256 mismatch. Verwacht $ExpectedZipSha256, kreeg $ActualZipSha256"
}

$Temp = Join-Path $Here ('.extract_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Temp -Force | Out-Null
try {
    Expand-Archive -Path $Zip -DestinationPath $Temp -Force
    Get-ChildItem -Path $Temp -Force | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $Here -Recurse -Force
    }
}
finally {
    if (Test-Path $Temp) {
        Remove-Item -Path $Temp -Recurse -Force
    }
}

$ManifestPath = Join-Path $Here 'SOURCE_MANIFEST_SHA256.json'
if (-not (Test-Path $ManifestPath)) {
    throw "Manifest ontbreekt na extractie: $ManifestPath"
}

$Manifest = Get-Content -Path $ManifestPath -Raw | ConvertFrom-Json -AsHashtable
$Failures = @()
foreach ($Entry in $Manifest.GetEnumerator()) {
    $Path = Join-Path $Here $Entry.Key
    if (-not (Test-Path $Path)) {
        $Failures += "MISSING $($Entry.Key)"
        continue
    }
    $Got = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Got -ne ([string]$Entry.Value).ToLowerInvariant()) {
        $Failures += "HASH $($Entry.Key): $Got"
    }
}
if ($Failures.Count -gt 0) {
    throw ("Source manifest verification failed:`n  " + ($Failures -join "`n  "))
}

if (-not (Test-Path $Python)) {
    throw "Python venv ontbreekt: $Python"
}

$PyFiles = Get-ChildItem -Path $Here -Filter '*.py' -File | Select-Object -ExpandProperty FullName
if ($PyFiles.Count -eq 0) {
    throw 'Geen Pythonbronbestanden gevonden na extractie.'
}
& $Python -m py_compile @PyFiles
if ($LASTEXITCODE -ne 0) {
    throw "Python syntaxcontrole faalde met return code $LASTEXITCODE"
}

Write-Host ''
Write-Host 'PRO-MAX V2 source installed and verified.' -ForegroundColor Green
Write-Host "Archive SHA-256 : $ActualZipSha256"
Write-Host "Verified files  : $($Manifest.Count)"
Write-Host ''
Write-Host 'Next commands:'
Write-Host '  .\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode install'
Write-Host '  .\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode smoke'
