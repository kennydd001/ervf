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

## V3 and V4 (2026-08-16)

G0/G1 above are the original, now-closed tracks (G0 failed parity, G1 failed
the no-regression gate — see `results/PRO_FINAL_REPORT.md`). V3 repaired the
identified causes and re-measured; V4 physically integrates both V3 wins into
one CUDA-graph capture. Run directly (no PowerShell wrapper needed for V4 yet):

```powershell
$env:LS_MODEL_DIR = 'nemotron_3_5_lightning_v35'
.\.venv-nemotron\Scripts\python.exe pro_research\graph_safe_v3.py --mode full
.\.venv-nemotron\Scripts\python.exe pro_research\selective_ervf_v3.py --mode full
.\.venv-nemotron\Scripts\python.exe pro_research\graph_selective_v4.py --mode full
```

Preregistrations: `PRO_V3_PREREGISTRATION.md`, `PRO_V4_PREREGISTRATION.md`.
Current best verified result: **41.13 tok/s** (V4, full mode, 765 samples) —
see `agents/RESEARCH_NOTEBOOK.md` for the full writeup, including why the
external V36/A1 anchor comparison is expected to diverge (model-identity note:
the anchor was frozen against a mislabeled Nemotron-3-Nano checkpoint, not the
true 3.5 Lightning model this pack targets).

G2 (`epoch_graph.py`) is technically blocked in its current form:
`cudaGraphLaunch()` on an already-instantiated graph is not itself capturable;
nesting a graph as a child node requires the pre-instantiation template plus
`cudaGraphAddChildGraphNode`, which `setup_graph()` does not currently retain.

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
