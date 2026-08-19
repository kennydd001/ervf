# S100 Phase 15 — native BF16 Tensor Cores with FP32 output

Date: 2026-08-19
Parent: `agent/s100-phase14r-repair-hardware@8944f31`

## Why Phase 14D does not close the native Tensor-Core hypothesis

Phase 14R validly removed WDDM paging, but its native candidate used a BF16 output tensor and then copied that rounded result into FP32. The current ERVF BF16 GEMV instead accumulates and writes FP32. Repeating an extra BF16 output quantization across 45,880 calls, including recurrent Mamba updates, is a different numerical model and can plausibly explain the catastrophic validation divergence.

The Phase-14R component benchmark also warmed each matrix independently four times and then timed eighteen repeats before moving to the next matrix. Summing all matrix sizes to claim >4x-L2 rotation does not make those per-matrix timings cold. Reported effective rates above 1 TB/s are therefore a cache-resident ceiling, not a real full-model weight-stream measurement.

14B2 and 14E2 remain closed. DFlash2 remains frozen closed for the current verifier.

## Candidate arithmetic

Use cuBLASLt/CUTLASS with:

- weight A/B storage: BF16 checkpoint bytes;
- activation input variants:
  1. BF16 input;
  2. FP32 input with supported fast-BF16 conversion where available;
- accumulator: FP32;
- output D: FP32;
- no BF16 output round-trip;
- no persistent transposed weight copy unless a separately admitted layout cache fits and its memory is included.

Candidate families:

1. `BF16_BF16_ACC32_OUT32` — primary;
2. `FP32_FAST_BF16_ACC32_OUT32` — optional library path;
3. `TF32_OUT32` — quality/control comparison;
4. current ERVF FP32-output baseline.

## Phase 15A — cold-stream ceiling

For B={1,2,4,8}:

- enumerate all live BF16 matrices;
- execute every matrix once in fixed layer order per repetition;
- only after the entire >4x-L2 stream may the next repetition begin;
- no per-matrix warmup/timing loops;
- preallocate all input/output workspaces;
- CUDA-event timing around the entire stream;
- report aggregate and family-level times;
- require measured DRAM/L2 behavior to be physically plausible;
- compare output against ERVF for every matrix.

Primary B=4 performance gate: >=2.5x useful-row throughput on the complete cold stream.

## Phase 15B — numerical attribution

Before full validation, run teacher-forced divergence diagnostics with variants:

- all BF16 matrices native;
- Mamba-in only;
- Mamba-out only;
- attention Q/K/V/O only by family;
- every Nth Mamba layer native;
- FP32-output candidate versus the old BF16-output candidate.

Record after every layer/token:

- hidden-state NRMSE/cosine;
- Mamba state NRMSE;
- convolution state NRMSE;
- route-id agreement;
- final logit KL/top-1;
- first layer/token crossing predefined error thresholds.

This distinguishes recurrent amplification from an integration/layout error.

## Phase 15C — official quality

Only a candidate with valid cold-stream performance proceeds:

1. strict `_02` validation;
2. frozen `_03/_04` heldout only after strict pass;
3. original official gates, deterministic repeat and finite outputs.

Family-selective promotion is allowed. A candidate does not need to replace every BF16 projection if only a safe subset provides useful latency.

## Phase 15D — block verifier

If B=1 quality is green and B=4 cold-stream speed is green, integrate the same FP32-output native kernels into the proven exact block-state/KV harness. Measure complete B=2/4/8 target verification, including state/KV commit and MoE work.

S100 requires full useful accepted-token wall time <=10 ms/token equivalent. Component speedups remain non-claims.

## Decisions

- `BF16_FP32_COLD_STREAM_OPEN`
- `BF16_FP32_QUALITY_OPEN`
- `BF16_FP32_FAMILY_SELECTIVE_OPEN`
- `NATIVE_BLOCK_VERIFIER_REOPEN`

A technical failure or cache-contaminated measurement remains `null`, not `false`.
