# S100 DualRHS execution addendum

Date: 2026-08-16
Branch: `pro-s100-dualrhs`
Status: frozen before the first DualRHS run.

The original microbenchmark imported `graph_e1f22._new_runtime`, which also pins the entire routed-expert bank and allocates cache state although this experiment touches only resident trunk/shared/router/attention/Mamba/LM-head weights. That extra bank is irrelevant to both candidate and reference arithmetic and can consume many GiB of host/pinned memory.

For the first scientific run, the PowerShell entry point therefore uses `s100_dualrhs_entry.py`, which replaces only the runtime constructor with a **lean resident shell**:

```text
LightningRuntime(model_dir, contexts_max=4096, embed_on_host=True,
                 fp8_kv=True, verbose=False)
```

It does not call `load_routed_bank()` or `enable_cache()`. No tested tensor, kernel, weight, activation, reference dispatch, candidate dispatch, timing loop, threshold or arithmetic rule changes.

For reproducibility, the runner also sets `PYTHONHASHSEED=0` before Python starts. The benchmark's name-derived RNG seeds are therefore stable across processes. This affects only synthetic activation generation; both timing arms still receive exactly the same activation pair.

This addendum does not relax or alter any gate in `S100_DUALRHS_ERVF_PREREGISTRATION.md`.
