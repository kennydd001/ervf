[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Repo
)
$ErrorActionPreference = "Stop"
$TargetBranch = "agent/s100-phase25-h8-best-of-all"
if (-not (Test-Path $Repo)) { throw "Repo not found: $Repo" }
$root = (& git -C $Repo rev-parse --show-toplevel).Trim()
$branch = (& git -C $root branch --show-current).Trim()
if ($branch -ne $TargetBranch) { throw "Refusing to publish from '$branch'; expected '$TargetBranch'." }

$paths = @(
    "pro_research/s100_phase25_*.py",
    "pro_research/S100_PHASE25_PREREGISTRATION.md",
    "pro_research/S100_PHASE25_PACK_MANIFEST_SHA256.json",
    "agents/S100_PHASE25_AGENT_HANDOFF.md",
    "RUN_ALL_S100_PHASE25.ps1",
    "PUBLISH_S100_PHASE25_RESULTS_GITHUB.ps1",
    "pro_research/results/s100_phase25",
    "reports/S100_PHASE25_RUN_REPORT.md"
)
foreach ($p in $paths) {
    & git -C $root add -- $p
    if ($LASTEXITCODE -ne 0) { throw "git add failed for $p" }
}

& git -C $root diff --cached --quiet
$hasChanges = ($LASTEXITCODE -ne 0)
if ($hasChanges) {
    & git -C $root commit -m "research: run S100 Phase 25 H8 best-of-all"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
} else {
    Write-Host "No staged Phase25 changes; nothing to commit."
}

& git -C $root push -u origin $TargetBranch
if ($LASTEXITCODE -ne 0) { throw "git push failed" }
$head = (& git -C $root rev-parse HEAD).Trim()
Write-Host "Published $TargetBranch at $head" -ForegroundColor Green
