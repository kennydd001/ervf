# S100 Phase 18 — Projection-block run report

Date: 2026-08-19  
Checkpoint: `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, snapshot `ce1b118ae66ec705d02c241525192832eb045fd3`  
Script: `pro_research/s100_phase18_projection_block.py`  
Result: `pro_research/results/s100_phase18_projection_block.json`

## Scope

This run isolates the projection bottleneck identified by Phase 17. It compares the current exact per-token projection with a block candidate at H=2/4/8 on real Mamba layers 0, 25, and 50. The candidate uses the existing packed NVFP4 representation or materialized BF16 weights. The run is deliberately bounded: it does not claim a complete Mamba-layer or B=4 verifier result.

The correctness gate is finite output plus NRMSE <= 1e-4 against the current production projection.

## Result

### NVFP4

Layers 0 and 50 are NVFP4 for the sampled in/out projections. The block candidate is bit-exact in every sampled case: NRMSE 0 and max-absolute error 0. It is nevertheless slower than the current production GEMV in the tested implementation.

Representative H=4 speedups:

| layer | side | speedup | correctness |
|---:|---|---:|---|
| 0 | in | 0.65x | pass |
| 0 | out | 0.85x | pass |
| 50 | in | 0.65x | pass |
| 50 | out | 0.81x | pass |

The best isolated NVFP4 sample was layer 50/out H=2 at roughly 0.95x, still not a repeatable win. This kernel is therefore not an opening for the full verifier; it is a correctness reference showing that the packed format can be consumed without drift, but not yet with a faster hardware path.

### BF16

Layer 25 is BF16 for the sampled in/out projections. Native `torch.mm` is substantially faster, but fails the production-output gate:

| side | H | BF16-output speedup | BF16-output NRMSE | FP32-output speedup | FP32-output NRMSE |
|---|---:|---:|---:|---:|---:|
| in | 4 | 5.33x | 2.37e-3 | 5.16x | 1.69e-3 |
| out | 4 | 4.49x | 2.88e-3 | 2.83x | 1.66e-3 |
| in | 8 | 9.81x | 2.33e-3 | 10.87x | 1.64e-3 |
| out | 8 | 4.49x | 2.88e-3 | 8.50x | 1.65e-3 |

The FP32-output contract improves the error, but does not reach the 1e-4 gate. Thus the speedup is real as a primitive measurement, but it cannot currently replace the exact production projection in a full Mamba block.

## Adjudication

`PHASE18_FULL_BLOCK_VERIFIER_OPEN = False`.

No full-layer/B=4 retest was opened because neither candidate satisfies the prerequisite combination of correctness and useful speed:

- NVFP4: correct, but slower.
- BF16: faster, but not correct enough against the production kernel.

This is a negative result for the tested projection implementations, not a falsification of all possible projection blocking. The next technically meaningful route is a real Tensor-Core/block-scaled projection implementation that preserves the intended accumulation contract, followed by an error-compensated or witness/fallback design for BF16 if the native activation quantization remains the dominant error source.

## Reproducibility

The complete machine-readable output is in `pro_research/results/s100_phase18_projection_block.json`. The runner completed with status `measured`; `all_correct` is false because the BF16 candidates fail the stated gate.
