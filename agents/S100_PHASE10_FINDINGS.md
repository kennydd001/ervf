# S100 phase 10 — hypothesis packs 10A/10B: final findings

Date: 2026-08-18 · Branch: agent/s100-phase10-hypotheses ·
Runners: `RUN_ALL_S100_PHASE10A.ps1` / `RUN_ALL_S100_PHASE10B.ps1`
(one-click master: `RUN_ALL_S100_PHASE10.ps1`).

Both preregistered hypothesis packs were executed end-to-end against the
completed phase-9 baseline. Every experiment ran to completion with valid
instrumentation; **neither hypothesis promoted**. All numbers below are read
from the result JSONs in `pro_research/results/s100_phase10a/` and
`pro_research/results/s100_phase10b/`.

## Environment fix (real error, minimal repair)

The 10B runner failed at start with `ModuleNotFoundError: No module named
'moe_lab'` — the pack venv `.venv-nemotron` had CuPy but not the editable
`moe-compiler-lab` install that `.venv` carries. Fixed with
`pip install -e . --no-deps` into `.venv-nemotron`; the suite then ran green.
(10A does not import `moe_lab` and was unaffected.)

## 10A — sparse routed-down panel cache: NOT promoted

Hypothesis: cache only the hot (layer, expert, panel) down-projection code
blocks (21,504 B each; scales stay in H-SCALE), sized 8–48 MiB by
calibration avoided-PCIe-bytes. Gates: bad-arm divergence, A/C/C/B parity,
<=1 ms drift, >=765 samples, >=0.15 ms/token gain.

Instrumentation validated: the bad arm diverges at every budget
(`bad_diverges: true` in smoke), and every candidate is bit-exact against
base (`cand_equals_base: true`, smoke and full).

Full A/C/C/B results (765+ samples per arm):

| Budget | base ms | cand ms | gain ms | cand tok/s | verdict |
|---|---|---|---|---|---|
| 8 MiB (390 panels)  | 18.593 | 19.408 | -0.815 | 51.52 | fail (gain gate only) |
| 16 MiB | 18.609 | 19.423 | -0.814 | 51.49 | fail (gain gate only) |
| 24 MiB | 18.606 | 19.418 | -0.813 | 51.50 | fail (gain gate only) |
| 32 MiB | 18.680 | 19.425 | -0.746 | 51.48 | fail (gain gate only) |
| 40 MiB | 18.631 | 19.428 | -0.797 | 51.47 | fail (gain gate only) |
| 48 MiB | 18.648 | 19.437 | -0.790 | 51.45 | fail (gain gate only) |

Interpretation: the penalty is essentially budget-independent
(-0.75..-0.82 ms/token from 8 to 48 MiB), so it is a fixed per-token kernel
overhead of the panel-cache GEMV path, not a miss-rate effect. Consistent
with the phase-9 oracle: total serial miss-fetch at the current 10.29% LRU
miss rate is ~1.5 ms/token and even perfect Belady (5.10%) caps the whole
miss-path win at ~0.75 ms — a panel sub-cache can only address a fraction of
that, while its gather/indirection costs ~0.8 ms/token regardless of hit
rate. `PANEL_CACHE_PROMOTE = False`.

## 10B — Mamba FP8 ERVF-v2 cold-stream bandwidth: NOT promoted

Hypothesis: a restructured FP8 GEMV (width 8/16/32 x default/cg/cs hints,
with/without L2 prefetch) beats the current `DenseERVF.mv_fp8_tensor` on the
real-weight cold stream over all 46 Mamba in/out matrices
(890,265,600 weight bytes per token equivalent). Selection gate: bit-exact
and >=5% faster on the stream; integration gate: >=0.25 ms/token end-to-end.

Cold-stream results (CUDA events, 24 reps, median; baseline = current ERVF):

| Variant | bit-exact | median ms | GB/s | speedup |
|---|---|---|---|---|
| baseline (ERVF) | — | 3.912 | 227.6 | 1.000 |
| w8_default      | yes | 6.118 | 145.5 | 0.639 |
| w8_pf_cs        | yes | 5.781 | 154.0 | 0.677 |
| w16_pf_default  | yes | 3.689 | 241.4 | **1.060** |
| w16_pf_cg       | yes | 4.041 | 220.3 | 0.968 |
| w16_pf_cs       | yes | 5.792 | 153.7 | 0.675 |
| w32_default     | yes | 5.524 | 161.2 | 0.708 |
| w32_pf_default  | yes | 4.160 | 214.0 | 0.940 |
| w32_pf_cg       | yes | 3.994 | 222.9 | 0.979 |
| w32_pf_cs       | yes | 3.927 | 226.7 | 0.996 |

Exactly one variant cleared selection: `w16_pf_default` (+6.0%, 241.4 GB/s).
Its integration A/C/C/B passed smoke (parity, drift) but the full 765-sample
compare gave base 18.491 ms vs cand 18.479 ms = **+0.012 ms/token**, far
below the 0.25 ms gate. `MAMBA_ERVF2_PROMOTE = False`.

Interpretation: the +0.22 ms stream win evaporates end-to-end — in the real
decode loop the Mamba GEMVs overlap with other work, so cold-stream
bandwidth is not the serial limiter at the current map. All nine variants
were bit-exact (max_abs 0.0); the current ERVF kernel is within 6% of the
best restructured variant measured.

## Updated bottleneck line

Phase 9 closed the cache/miss path; phase 10 now also closes:

- **panel-level weight sub-caching** (10A): fixed overhead exceeds any
  reachable miss saving;
- **Mamba FP8 streaming-kernel restructuring** (10B): +6% on the isolated
  cold stream, +0.012 ms/token integrated — not a serial bottleneck.

Best measured config is unchanged: current map at ~18.5–18.6 ms/token
(~54 tok/s). S100 single not achieved. The remaining first-principles
lever, per the phase-9 roofline (dense GEMV at 37.8% of 338.4 GB/s) now
confirmed by direct measurement (227–241 GB/s practical stream ceiling on
this GPU), is **reducing dense bytes per token** (weight format/structure),
not fetching or streaming them faster.
