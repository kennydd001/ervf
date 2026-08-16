# PRO-MAX V2 — post-V6 exact campaign

This branch starts from the latest agent state and prunes the old P000-P099
research tree before adding new work. The exact disposition is in
`P000_P099_STATUS.md`.

Current verified record:

```text
21.0923 ms/token = 47.4107 tok/s
```

Only `1.0923 ms/token` remains to 50 tok/s. This pack therefore prioritises
three exact graph final-mile candidates and one low-level graph-amortisation
probe. It does not restart already closed work.

## Run

From the repository root in PowerShell 7:

```powershell
git fetch origin
git switch pro-max-v2
git pull --ff-only origin pro-max-v2

.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode install
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode smoke
```

After a clean smoke run:

```powershell
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode full
```

Architecture-only probes can be run separately:

```powershell
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode architecture
```

All outputs stay under:

```text
pro_research/results/pro_max_v2/
```

The campaign never kills another CUDA process. Close other LLM runtimes before
starting. Keep the laptop on AC power and disable sleep for the full run.

## Expected result files

```text
PV2_00_PROVENANCE.json
PV2_10_ADDNORM.json
PV2_11_QKV.json
PV2_12_LMHEAD_ARGMAX.json
PV2_13_FINALE.json
PV2_20_CHILD_EPOCH.json
PV2_21_CAPABILITIES.json
PV2_VERIFICATION.json
PV2_FINAL_REPORT.md
```

No GPU result is claimed by this commit. The Python source is syntax-checked in
install mode; the CUDA kernels must be compiled and measured on the target
Blackwell laptop.
