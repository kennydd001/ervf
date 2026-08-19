param([string]$Destination='')
$ErrorActionPreference='Stop'
$Repo=(& git rev-parse --show-toplevel 2>$null).Trim()
if(-not $Repo){throw 'Run inside the ERVF worktree.'}
if(-not $Destination){$Destination=$Repo}
$Here=$PSScriptRoot
$Manifest=Get-Content (Join-Path $Here 'PACK_MANIFEST.json') -Raw|ConvertFrom-Json
$Builder=[System.Text.StringBuilder]::new()
foreach($Part in $Manifest.parts){
 $Path=Join-Path $Here $Part.name
 if(-not(Test-Path $Path)){throw "Missing pack part: $Path"}
 $Actual=(Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
 if($Actual -ne $Part.sha256){throw "Part hash mismatch: $($Part.name)"}
 [void]$Builder.Append((Get-Content $Path -Raw).Trim())
}
$Bytes=[Convert]::FromBase64String($Builder.ToString())
$Tmp=Join-Path $env:TEMP $Manifest.zip_name
[IO.File]::WriteAllBytes($Tmp,$Bytes)
$ZipHash=(Get-FileHash $Tmp -Algorithm SHA256).Hash.ToLowerInvariant()
if($ZipHash -ne $Manifest.zip_sha256){throw "ZIP hash mismatch: $ZipHash"}
Expand-Archive $Tmp $Destination -Force
Write-Host "Migration source extracted to: $Destination" -ForegroundColor Green
Write-Host 'Next: run RUN_ALL_S100_LIGHTNING_MIGRATION_AUDIT.ps1.' -ForegroundColor Green
