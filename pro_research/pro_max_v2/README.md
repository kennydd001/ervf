# PRO-MAX V2 — post-V6 exact campaign

This branch starts from `pro-research@5c699300` and prunes the original P000–P099 research tree before opening new work.

Current verified record:

```text
21.0923 ms/token = 47.4107 tok/s
```

Only `1.0923 ms/token` remains to 50 tok/s. The source pack tests three exact final-mile candidates and one low-level CUDA child-graph route:

- residual add + next RMSNorm;
- mixed-shape exact Q/K/V in one launch;
- LM-head ERVF + exact hierarchical greedy argmax;
- physical composition on top of V6;
- exact K-token child-graph epochs using `cudaGraphAddChildGraphNode`, not the previously failed nested-capture approach.

The human-readable source is stored as seven ordered Base64 payload parts because the GitHub connector cannot safely upload this binary ZIP in one call. `INSTALL_SOURCE.ps1` reconstructs the archive, verifies its fixed SHA-256, extracts it and verifies every source file against `SOURCE_MANIFEST_SHA256.json`.

## Checkout and install

From the repository root in PowerShell 7:

```powershell
git fetch origin
git switch pro-max-v2
git pull --ff-only origin pro-max-v2
.\pro_research\pro_max_v2\INSTALL_SOURCE.ps1
```

Then run the CPU/preflight checks and technical GPU smoke campaign:

```powershell
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode install
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode smoke
```

After a clean smoke run:

```powershell
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode full
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode architecture
```

Or one complete unattended sequence:

```powershell
.\pro_research\pro_max_v2\RUN_POST_V6.ps1 -Mode overnight
```

All outputs stay under:

```text
pro_research/results/pro_max_v2/
```

Push only those results with:

```powershell
.\pro_research\pro_max_v2\PUSH_RESULTS.ps1
```

No GPU result is claimed by this commit. The Python source was syntax-checked; the CUDA candidates must compile and run on the target Blackwell laptop.
