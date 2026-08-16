$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Branch = (git branch --show-current).Trim()
if ($Branch -ne 'pro-s100-nativefp4-c2b') {
    throw "Refusing push from branch '$Branch'; expected pro-s100-nativefp4-c2b"
}
$Result = 'pro_research/results/native_nvfp4/C2B_TORCH212_CONTRACT.json'
$Verify = 'pro_research/results/native_nvfp4/C2B_TORCH212_CONTRACT_VERIFICATION.json'
if (-not (Test-Path $Result)) { throw "Missing C2b result: $Result" }
if (-not (Test-Path $Verify)) { throw "Missing C2b verification: $Verify" }

# Fail closed: only the two C2b result artifacts are staged by this helper.
git add -f -- $Result $Verify
$staged = git diff --cached --name-only
if (-not $staged) { throw 'No staged C2b result changes.' }
$unexpected = @($staged | Where-Object { $_ -notin @($Result.Replace('\','/'), $Verify.Replace('\','/')) })
if ($unexpected.Count -gt 0) { throw "Unexpected staged paths: $($unexpected -join ', ')" }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --cached --check failed' }

git commit -m 'S100 native NVFP4 C2b isolated Torch 2.12 results'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git push origin pro-s100-nativefp4-c2b
if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
Write-Host 'C2b native NVFP4 results pushed.' -ForegroundColor Green
