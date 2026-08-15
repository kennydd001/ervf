# PRO research pack

Additive breakthrough experiments for the current Nemotron 3.5 Lightning / ERVF
runtime. The pack is based on commit
`96811c4e381bf788f9133f5d1fc025e6885cf78f` and is intentionally isolated from
all closed research namespaces.

## What changed after the latest Kimi work

The major new fact is that the full-token CUDA graph is already built in
`runtime.py`, and device-resident routing/cache has already produced a measured
end-to-end gain. What was missing was the required graph A/B runner and verifier.
This pack fills that gap first.

It then tests two derived, high-upside hypotheses:

1. apply ERVF's exact virtual reduction to the resident BF16, FP8-tensor and
   FP32 GEMVs that still use the old 256-thread-per-row geometry;
2. capture K causal token-graph replays into one exact parent graph, so one host
   launch advances several autoregressive tokens without speculation.

Read [`PRO_HYPOTHESES.md`](PRO_HYPOTHESES.md) for the reasoning and gates.

## Safe branch

The files are committed on branch:

```text
pro-research
```

No existing runtime/report file is modified. Results go only to:

```text
pro_research/results/
```

An existing result is moved to `results/history/` before a replacement is
written.

## First checkout

```powershell
git fetch origin
git switch pro-research
.\pro_research\INSTALL_AND_RUN.ps1 -Mode install
```

## Recommended sequence

Technical smoke:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke
```

Then inspect:

```text
pro_research/results/PRO_FINAL_REPORT.md
pro_research/results/PRO_VERIFICATION.json
```

Full run:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode full
```

Individual tracks:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode graph
.\pro_research\INSTALL_AND_RUN.ps1 -Mode dense
.\pro_research\INSTALL_AND_RUN.ps1 -Mode epoch
```

Rebuild only the verifier/report:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode verify
.\pro_research\INSTALL_AND_RUN.ps1 -Mode report
```

## Files

| file | purpose |
|---|---|
| `graph_e1f22.py` | frozen EGR/GRAPH/CTL/DET full-token graph A/B |
| `ervf_dense.py` | exact ERVF kernels for BF16, FP8-tensor and FP32 GEMV + A/B/A integration |
| `epoch_graph.py` | exact K-token parent-graph probe |
| `verify_results.py` | independent CPU-side gate recomputation; imports no runner |
| `build_report.py` | plain-language Dutch report |
| `run_all.py` | sequential dependency-aware orchestrator |
| `common.py` | result archiving, hashes, environment and GPU-free checks |
| `EXPERIMENT_REGISTRY.yaml` | frozen PRO gates |

## Important boundaries

- A microbenchmark is never reported as tok/s.
- Exactness gates are not relaxed after results.
- The scripts never kill another GPU process.
- Gatherless downflow, low-rank ReLU2 prediction and the closed speculative
  branch are not reopened.
- The product-level breakthrough remains an independently verified integrated
  causal run of at least 50 tok/s, not merely another fast component.

The pack was syntax-checked and its CPU reduction-tree equivalence selftest was
run before commit. GPU performance and graph support must be established by the
manual target-hardware runs.
