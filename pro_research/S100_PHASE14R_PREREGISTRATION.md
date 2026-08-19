# S100 Phase 14R preregistration

Date: 2026-08-19

## 14D-ZC component

- Use `LightningRuntime`, not the resident QFAST cache parent.
- Import Torch before CuPy/runtime construction.
- DLPack-alias existing BF16 weights; no clone and no contiguous transpose.
- Call `torch.mm(X_bf16, W_bf16.T, out=preallocated_output)`.
- Measure with CUDA events on one shared stream.
- B={2,4,8}; all live BF16 matrices; real-weight rotation >4x L2.
- Correctness against current ERVF.
- B=4 gate: useful-row speedup >=2.5x, max case NRMSE <=0.005,
  mean row-argmax agreement >=0.97.
- Paging validity: aggregate native effective bandwidth >=40 GB/s and no
  >=16 MiB matrix below 20 GB/s.
- Log free VRAM before and after each case.

## 14D-ZC quality

Only after the valid B=4 component gate:

- current QFAST + alpha=0.0003 quality parent;
- eager execution;
- normalized MoE return contract;
- zero-copy BF16 weight aliases;
- preallocated input/output conversion buffers;
- B=1 native GEMM numerical path;
- strict `_02` validation;
- frozen `_03/_04` heldout only after strict validation.

## 14B2

- calibration-only reduced-rank fit;
- BF16-rounded factors;
- validation output metrics;
- ranks 32/64/128/192/256/384;
- physical byte saving >=35%;
- output NRMSE <=0.03;
- cosine >=0.9995;
- p95 relative row error <=0.08;
- a family opens at >=80% passing representative cases.

## 14E2

- decoded real NVFP4 expert samples;
- real repaired MoE input/activation/route captures;
- activation-weighted expert-axis basis;
- ranks 4/8/16/32;
- exact override residual blocks 6.25/12.5/25%;
- byte ratio <=0.70;
- sampled validation GEMV NRMSE <=0.05;
- cosine >=0.999 on all sampled layers.

## Claims

DFlash2 flags remain frozen false. These tests only open later runtime builds.
S100 remains unclaimed.
