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

The complete human-readable source is stored in `PRO_MAX_V2_SOURCE.zip` because this branch was assembled through the GitHub connector. Extract it once:

```powershell
.\pro_research\pro_max_v2\INSTALL_SOURCE.ps1
```

Then run:

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

All outputs stay under `pro_research/results/pro_max_v2/`. No GPU result is claimed by this commit; the Python source was syntax-checked, but the CUDA candidates must compile and run on the target Blackwell laptop.
