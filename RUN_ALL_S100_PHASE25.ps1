[CmdletBinding()]
param(
    [switch]$Publish,
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$PackRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentBranch = "agent/s100-phase24-best-of-all"
$TargetBranch = "agent/s100-phase25-h8-best-of-all"

function Test-ERVFRepo([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path)) { return $false }
    if (-not (Test-Path (Join-Path $Path "pro_research\s100_phase24_common.py"))) { return $false }
    if (-not (Test-Path (Join-Path $Path "pro_research\results\s100_phase24\S100_PHASE24_SUMMARY.json"))) { return $false }
    try {
        $root = (& git -C $Path rev-parse --show-toplevel 2>$null).Trim()
        return [bool]$root
    } catch { return $false }
}

function Find-ERVFRepo {
    if ($Repo -and (Test-ERVFRepo $Repo)) {
        return (& git -C $Repo rev-parse --show-toplevel).Trim()
    }
    if ($env:ERVF_REPO -and (Test-ERVFRepo $env:ERVF_REPO)) {
        return (& git -C $env:ERVF_REPO rev-parse --show-toplevel).Trim()
    }
    $cwd = (Get-Location).Path
    if (Test-ERVFRepo $cwd) { return (& git -C $cwd rev-parse --show-toplevel).Trim() }

    $roots = @(
        (Join-Path $env:USERPROFILE "Documents\ChatGPT"),
        (Join-Path $env:USERPROFILE "Documents")
    ) | Select-Object -Unique
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        Write-Host "Searching ERVF repo under $root ..." -ForegroundColor DarkGray
        $hits = Get-ChildItem -Path $root -Filter "s100_phase24_common.py" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '[\\/]pro_research[\\/]s100_phase24_common\.py$' } |
            Select-Object -First 12
        foreach ($hit in $hits) {
            $candidate = Split-Path -Parent (Split-Path -Parent $hit.FullName)
            if (Test-ERVFRepo $candidate) {
                return (& git -C $candidate rev-parse --show-toplevel).Trim()
            }
        }
        if ($hits) { break }
    }
    throw "Could not find an ERVF worktree containing the published Phase24 parent. Set `$env:ERVF_REPO explicitly."
}

function Find-WorktreeForBranch([string]$SourceRepo,[string]$Branch) {
    $want = "refs/heads/$Branch"
    $lines = & git -C $SourceRepo worktree list --porcelain
    $path = $null
    foreach ($line in $lines) {
        if ($line -like "worktree *") { $path = $line.Substring(9) }
        elseif ($line -eq "branch $want" -and $path) { return $path }
        elseif ([string]::IsNullOrWhiteSpace($line)) { $path = $null }
    }
    return $null
}

function Ensure-Phase25Worktree([string]$SourceRepo) {
    $existing = Find-WorktreeForBranch $SourceRepo $TargetBranch
    if ($existing -and (Test-Path $existing)) { return $existing }

    try {
        & git -C $SourceRepo fetch origin $ParentBranch --quiet
        if ($LASTEXITCODE -ne 0) { throw "fetch exit $LASTEXITCODE" }
    } catch {
        Write-Warning "Could not refresh origin/$ParentBranch; using an existing local parent ref if available."
    }

    $parentRef = $null
    & git -C $SourceRepo show-ref --verify --quiet "refs/remotes/origin/$ParentBranch"
    if ($LASTEXITCODE -eq 0) { $parentRef = "origin/$ParentBranch" }
    if (-not $parentRef) {
        & git -C $SourceRepo show-ref --verify --quiet "refs/heads/$ParentBranch"
        if ($LASTEXITCODE -eq 0) { $parentRef = $ParentBranch }
    }
    if (-not $parentRef) { throw "Phase24 parent ref $ParentBranch is not available locally or on origin." }

    $parentDir = Split-Path -Parent $SourceRepo
    $target = Join-Path $parentDir "ervf-s100-phase25-h8-best-of-all"
    if (Test-Path $target) {
        if (Test-ERVFRepo $target) {
            $b = (& git -C $target branch --show-current).Trim()
            if ($b -eq $TargetBranch) { return $target }
        }
        throw "Target worktree directory already exists but is not the expected Phase25 branch: $target"
    }

    & git -C $SourceRepo show-ref --verify --quiet "refs/heads/$TargetBranch"
    if ($LASTEXITCODE -eq 0) {
        $null = & git -C $SourceRepo worktree add $target $TargetBranch
    } else {
        $null = & git -C $SourceRepo worktree add -b $TargetBranch $target $parentRef
    }
    if ($LASTEXITCODE -ne 0) { throw "git worktree add failed" }
    return $target
}

function Copy-Overlay([string]$TargetRepo) {
    New-Item -ItemType Directory -Force -Path (Join-Path $TargetRepo "pro_research") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $TargetRepo "agents") | Out-Null
    Get-ChildItem (Join-Path $PackRoot "pro_research") -File | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $TargetRepo "pro_research\$($_.Name)") -Force
    }
    Copy-Item (Join-Path $PackRoot "agents\S100_PHASE25_AGENT_HANDOFF.md") (Join-Path $TargetRepo "agents\S100_PHASE25_AGENT_HANDOFF.md") -Force
    Copy-Item (Join-Path $PackRoot "RUN_ALL_S100_PHASE25.ps1") (Join-Path $TargetRepo "RUN_ALL_S100_PHASE25.ps1") -Force
    Copy-Item (Join-Path $PackRoot "PUBLISH_S100_PHASE25_RESULTS_GITHUB.ps1") (Join-Path $TargetRepo "PUBLISH_S100_PHASE25_RESULTS_GITHUB.ps1") -Force
    if (Test-Path (Join-Path $PackRoot "MANIFEST_SHA256.json")) {
        Copy-Item (Join-Path $PackRoot "MANIFEST_SHA256.json") (Join-Path $TargetRepo "pro_research\S100_PHASE25_PACK_MANIFEST_SHA256.json") -Force
    }
}

function Resolve-Python([string]$TargetRepo,[string]$SourceRepo) {
    $candidates = @(
        (Join-Path $TargetRepo ".venv\Scripts\python.exe"),
        (Join-Path $SourceRepo ".venv\Scripts\python.exe"),
        (Join-Path $SourceRepo "venv\Scripts\python.exe")
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    $venvs = Get-ChildItem -Path $SourceRepo -Directory -Filter ".venv*" -ErrorAction SilentlyContinue
    foreach ($v in $venvs) {
        $p = Join-Path $v.FullName "Scripts\python.exe"
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "No Python interpreter found. Activate the same environment used for Phase24 or set it on PATH."
}

function Invoke-Py([string]$Name,[string[]]$Arguments,[switch]$AllowFail) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $script:Python @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        if ($AllowFail) { Write-Warning "$Name exited with code $code; campaign continues fail-closed." }
        else { throw "$Name failed with exit code $code" }
    }
    return $code
}

$SourceRepo = Find-ERVFRepo
$Phase25Repo = Ensure-Phase25Worktree $SourceRepo
Copy-Overlay $Phase25Repo
Set-Content -Path (Join-Path $PackRoot "LAST_PHASE25_WORKTREE.txt") -Value $Phase25Repo -Encoding UTF8
$Python = Resolve-Python $Phase25Repo $SourceRepo
$env:PYTHONPATH = "$Phase25Repo\pro_research;$Phase25Repo;$env:PYTHONPATH"

Write-Host "Source Phase24 repo: $SourceRepo" -ForegroundColor Green
Write-Host "Phase25 worktree:    $Phase25Repo" -ForegroundColor Green
Write-Host "Python:              $Python" -ForegroundColor Green

Push-Location $Phase25Repo
try {
    Invoke-Py "Phase25 preflight" @("pro_research\s100_phase25_preflight.py")

    Invoke-Py "Fresh Phase24 H4+H4 parent screen" @(
        "pro_research\s100_phase25_measure.py","--arm","parent","--context","1024","--tag","screen","--blocks","8","--warmup","4"
    )

    $variants = @("split4_route","direct8_route","direct8_groupdown")
    foreach ($v in $variants) {
        Invoke-Py "H8 screen $v" @(
            "pro_research\s100_phase25_measure.py","--arm","candidate","--variant",$v,"--context","1024","--tag","screen","--blocks","8","--warmup","4"
        ) -AllowFail | Out-Null
    }

    Invoke-Py "Phase24 parent full-state capture over 8 tokens" @(
        "pro_research\s100_phase25_state_capture.py","--mode","parent"
    )

    foreach ($v in $variants) {
        $screen = Join-Path $Phase25Repo "pro_research\results\s100_phase25\S100_PHASE25_SCREEN_$($v.ToUpper())_CTX1024.json"
        if (-not (Test-Path $screen)) { continue }
        $meta = Get-Content $screen -Raw | ConvertFrom-Json
        if ($meta.status -ne "measured") { continue }
        $code = Invoke-Py "Full-state capture $v" @(
            "pro_research\s100_phase25_state_capture.py","--mode","candidate","--variant",$v
        ) -AllowFail
        if ($code -eq 0) {
            Invoke-Py "Full-state compare $v" @(
                "pro_research\s100_phase25_state_compare.py","--variant",$v
            ) -AllowFail | Out-Null
        }
    }

    Invoke-Py "H8 exact/state-green selection" @("pro_research\s100_phase25_select.py")
    $selectionPath = Join-Path $Phase25Repo "pro_research\results\s100_phase25\S100_PHASE25_SELECTION.json"
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json

    if ($selection.selected) {
        $winner = [string]$selection.selected.variant
        Invoke-Py "Selected H8 context 128" @(
            "pro_research\s100_phase25_measure.py","--arm","candidate","--variant",$winner,"--context","128","--tag","promoted","--blocks","8","--warmup","4"
        ) -AllowFail | Out-Null
        Invoke-Py "Selected H8 context 4096" @(
            "pro_research\s100_phase25_measure.py","--arm","candidate","--variant",$winner,"--context","4096","--tag","promoted","--blocks","4","--warmup","2"
        ) -AllowFail | Out-Null
        Invoke-Py "Selected H8 synchronized bottleneck profile" @("pro_research\s100_phase25_profile.py") -AllowFail | Out-Null
    }

    if ($selection.THERMAL_ADOPTION_OPEN) {
        $rounds = @(
            @("R1_PARENT","parent"), @("R1_SELECTED","selected"),
            @("R2_SELECTED","selected"), @("R2_PARENT","parent"),
            @("R3_PARENT","parent"), @("R3_SELECTED","selected"),
            @("R4_SELECTED","selected"), @("R4_PARENT","parent")
        )
        foreach ($r in $rounds) {
            Invoke-Py "Thermal $($r[0])" @(
                "pro_research\s100_phase25_thermal_measure.py","--mode",$r[1],"--context","1024","--tag",$r[0],"--blocks","16","--warmup","8"
            ) -AllowFail | Out-Null
        }
        Invoke-Py "Phase25 thermal adjudication" @("pro_research\s100_phase25_thermal_adjudicate.py") -AllowFail | Out-Null
    } else {
        Write-Host "Thermal adoption not opened by preregistered screen gates; skipping thermal rounds." -ForegroundColor Yellow
    }

    Invoke-Py "Phase25 summary" @("pro_research\s100_phase25_summary.py")
    Invoke-Py "Phase25 markdown report" @("pro_research\s100_phase25_report.py")

    $summaryPath = Join-Path $Phase25Repo "pro_research\results\s100_phase25\S100_PHASE25_SUMMARY.json"
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Host "`n=== PHASE 25 FINAL ===" -ForegroundColor Magenta
    Write-Host "Selected variant:          $($summary.selected_variant)"
    Write-Host "State green:               $($summary.state_green)"
    Write-Host "H8 adopted:                $($summary.gates.H8_ADOPTED)"
    Write-Host "S100 target-only achieved: $($summary.S100_TARGET_ONLY_ACHIEVED)"
    Write-Host "S100 single achieved:      $($summary.S100_SINGLE_ACHIEVED)"
    Write-Host "Next route:                $($summary.NEXT_ROUTE)"
    Write-Host "Summary: $summaryPath"
    Write-Host "Report:  $(Join-Path $Phase25Repo 'reports\S100_PHASE25_RUN_REPORT.md')"

    if ($Publish) {
        & (Join-Path $PackRoot "PUBLISH_S100_PHASE25_RESULTS_GITHUB.ps1") -Repo $Phase25Repo
        if ($LASTEXITCODE -ne 0) { throw "Phase25 publication failed" }
    } else {
        Write-Host "`nGitHub publication was not requested. Re-run with -Publish after inspection." -ForegroundColor DarkGray
    }
}
finally {
    Pop-Location
}
