# PRO-MAX V2 result intake — 2026-08-16

Source: user-uploaded `pro_max_v2(1).zip`, read after the full and smoke runs.
This file records the raw conclusions needed for the next branch; it does not
reinterpret failed preregistered gates as passes.

## Exact final-mile candidates

### PV2-10 Add + next RMSNorm

- micro hidden output: bitexact
- micro normalized output: bitexact
- micro speed gate: pass
- graph structural gate: pass
- full causal parity: **FAIL** — one prompt first diverges at generated token 124
- BASE_A / candidate / BASE_B p50: `20.5737 / 21.3962 / 22.9070 ms`
- baseline drift: `2.3333 ms` — invalidates a clean performance attribution
- verdict: **do not adopt**; the late causal divergence requires isolation before any retry.

### PV2-11 mixed Q/K/V one-launch

- micro Q/K/V: bitexact
- micro speed gate: pass
- graph structural gate: pass
- full causal parity: **PASS on all three prompts**
- BASE_A / candidate / BASE_B p50: `20.7964 / 21.4866 / 22.6541 ms`
- baseline midpoint: `21.72525 ms`; candidate is `0.23865 ms` below it
- baseline drift: `1.8577 ms` — preregistered drift gate fails
- verdict: correctness survives; performance is **unresolved**, not negative. Re-measure only under a steady-state/interleaved harness.

### PV2-12 LM-head + hierarchical exact argmax

- every debug logit: bitexact
- top-1: exact
- causal parity: pass
- micro speed gate: fail
- full no-regression gate: fail
- BASE_A / candidate / BASE_B p50: `20.7965 / 22.3746 / 22.9191 ms`
- verdict: **negative for this implementation**; do not adopt.

### PV2-13 finale

No candidate passed the frozen adoption rule, so V10 intentionally selected
none. Causal/determinism/control gates still behaved correctly. The run failed
its adoption and baseline-drift gates; it is not a new performance baseline.

## PV2-20 exact child-graph result — key new finding

The low-level child graph itself did **not** beat a set of individually queued
child launches:

- K=2: separate queued `18.7580 ms/token`, parent `19.4709`, exact ids
- K=4: separate queued `19.0660 ms/token`, parent `19.3495`, exact ids

Thus `cudaGraphAddChildGraphNode` closes the specific "one parent graph is
faster" hypothesis for this implementation.

However, the **control arm is itself a new systems result**: the exact same
autoregressive single-sequence token graph runs below 20 ms/token when several
replays are queued before the host performs a blocking harvest. The production
V6 measurement instead calls `ring_harvest()` after every token, and that method
synchronizes the graph stream.

This suggests roughly two milliseconds of the observed V6 token interval are
host synchronization / queue-starvation rather than model arithmetic. The
next experiment must distinguish three metrics explicitly:

1. synchronous request/response token latency;
2. single-sequence queued generation throughput;
3. individually streamed host-delivery gaps while the GPU queue stays full.

Only (3), if >=50 tok/s with exact tokens and bounded delivery gaps, justifies a
strong interactive/streaming E50 claim. Metric (2) may legitimately be called
single-sequence generation throughput but not per-token host round-trip latency.

## Capability result

Target stack reports CUDA runtime 12.9, driver runtime 13.2 interface,
compute capability `12.0` / `sm_120`. `cudart64_12.dll` exports
`cudaGraphAddChildGraphNode`, `cudaGraphExecUpdate`, conditional-handle APIs and
the normal graph API. `nvcc`, `nsys`, `ncu` and `compute-sanitizer` were not on
PATH. TMA architecture prerequisite is present; mapped-host-to-SMEM TMA remains
unproven and requires a physical byte-exact microbenchmark.

## Next experiment

`PRO V12`: keep V6 arithmetic unchanged, preheat to steady state, compare
SYNC_A / queued K={2,4,8,16,32} / event-streamed K / SYNC_B. Tokens are chained
only through `_tok_dev`; the host is not needed to choose the next token. The
existing pinned ring D2H copy remains. Reusable non-timing CUDA events are
recorded after each D2H copy so the host can consume individual ring slots
without synchronizing the whole stream.

E50 subclaims are frozen separately:

- queued E50: exact >=50 generated tok/s on one sequence with batch harvest;
- streamed E50: exact >=50 host-delivered tok/s with p50 inter-delivery gap
  <=20 ms after queue warmup;
- synchronous E50: current blocking per-token arm <=20 ms p50.
