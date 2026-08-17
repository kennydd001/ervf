
# S100 phase 7 decision — recover the censored frontier and pack downflow

Date: 2026-08-17

## Phase-6 result

The fastest fully measured quality-green arm is:

- QFAST + relative routed-down threshold alpha 0.0003;
- 18.6276 ms/token;
- 53.6838 tok/s;
- 5,120 heldout target tokens;
- top1 agreement 0.980859;
- target recall in top5 1.0;
- mean CE delta 0.008427;
- mean coarse KL 0.008640.

S100-single remains false.

## Why eight phase-6 results are not quality failures

Phase 6 evaluated all heldout candidates inside one Python/CUDA process. The
first candidate completed. A later illegal memory access poisoned the CUDA
context, after which module lookup and ordinary GEMV launches failed for every
remaining candidate.

The affected results are censored technical failures. They do not establish
that alpha 0.0010, alpha 0.0015, K1, K2 or the compositions fail fidelity.

Phase 7 preserves the exact phase-6 candidate set and starts one fresh process
for every candidate. It does not add or remove a candidate after heldout data
was visible.

## Exact backend decision

- direct mapped-host down GEMV is closed: approximately 27.3 ms versus 18.8 ms;
- direct plus ballot/fusion is closed: approximately 26.7 ms;
- ballot plus fused reduction was exact but neutral in full mode.

Phase 7 therefore does not retry direct host compute.

## New exact hypothesis: packed sparse mirror

The legacy gather copies each selected code column to its original offset in a
2.806 MB panel-major mirror. The following down GEMV revisits those sparse
offsets.

The packed backend instead:

1. preserves the ascending selected-column list;
2. gathers selected code columns contiguously;
3. records the packed start offset of every active panel;
4. executes the same panel-to-chunk assignment;
5. visits mask bits in the same ascending order;
6. reads the same resident scale byte and activation;
7. uses the same fmaf sequence and unchanged reduction.

This removes sparse destination addressing and improves locality without
changing bytes crossing PCIe or arithmetic. Exact token parity is mandatory.

## Large route after phase 7

Even a successful phase-7 frontier remains more than eight milliseconds from
100 single-stream tok/s. Small exact gains are retained, but the next
architectural gate is a real SM120 grouped-MoE experiment.

CUTLASS 4.6 contains SM120 NVFP4 grouped kernels, although the generator path
has a known gap for grouped NVFP4 emission. The concrete implementation route is
therefore the known-working SM120 grouped example/template, not blind reliance
on profiler generation.
