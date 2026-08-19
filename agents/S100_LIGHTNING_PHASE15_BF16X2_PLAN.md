# S100 Lightning Phase 15 — provenance reset + BF16x2 Tensor Core path

Date: 2026-08-19
Parent: `751f546` on `agent/s100-phase14r-repair-hardware`
Target checkpoint: NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 only.

## Material provenance correction

Nano and Lightning are shape-identical except for the declared context ceiling, but their weights, activations, routing, logits and quality traces are different. Shape/kernel work may transfer; model-dependent quality, route, subspace, expert-basis, speculative-acceptance and frozen-trace conclusions do not.

Every Phase-15 artifact must include a checkpoint identity record and refuse a model whose `max_position_embeddings` is not 1,048,576. No Nano `S100_PHASE3_V18_TRACE_FULL` may be used as a Lightning quality reference.

## First falsification

Run the exact Lightning quality parent against the inherited trace. If agreement is not effectively 100%, quarantine that trace and create a fresh Lightning-parent trace. All later calibration/validation/heldout comparisons use only the new trace.

## Why 14D remains promising

The paging-free Lightning component measured B=4 native BF16 matmul about 7.4x faster than independent ERVF rows. The quality harness, however, rounded both the activation input and every projection output to BF16. It also compared against an inherited trace whose checkpoint provenance was not enforced.

## Precision ladder

For attention K/V/O, test:

1. current ERVF — FP32 activation and FP32 output;
2. BF16-rounded activation through current ERVF — isolates input rounding;
3. one-term native BF16 Tensor Core with FP32 accumulation/output;
4. two-term BF16 split (`x = x_hi + x_lo`) in one multi-row GEMM, FP32 output and FP32 sum;
5. three-term BF16 split control.

The two-term form is the main hypothesis. At B=4 it becomes one M=8 GEMM, so the same weight tile can serve four high and four residual rows. This may recover near-FP32 activation fidelity while preserving most B=4 weight reuse.

## Family attribution

Evaluate K, V, O, K+V and K+V+O separately on Lightning calibration. If a full replacement fails, family-selective and then layer-selective promotion is allowed.

## Performance protocol

Cold stream means every live BF16 matrix is executed once in fixed layer order before any repetition. No per-matrix warmup/timing loop. Inputs, outputs and split buffers are preallocated. B={1,2,4,8}; terms={1,2,3}. Report full-stream CUDA-event time and physically plausible effective bandwidth.

## Quality protocol

1. fresh Lightning parent trace;
2. parent self-control must be exact;
3. calibration selection;
4. strict validation;
5. heldout only after strict pass;
6. deterministic repeat and finite output.

## Reopened downstream work

Nano-derived Phase-12/14F verifier and DFlash2 numbers are not Lightning evidence. They remain quarantined until the Lightning verifier floor and acceptance proxy are rerun. A quality-green BF16x2 candidate is integrated into that rerun first.

## Decisions

- `LIGHTNING_TRACE_PROVENANCE_GREEN`
- `BF16X2_COLD_STREAM_OPEN`
- `BF16X2_QUALITY_OPEN`
- `BF16X2_FAMILY_SELECTIVE_OPEN`
- `LIGHTNING_BLOCK_VERIFIER_RERUN_OPEN`

No component or inherited Nano result can claim S100 for Lightning.
