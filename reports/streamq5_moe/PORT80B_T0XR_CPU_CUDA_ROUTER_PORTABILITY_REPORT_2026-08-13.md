# PORT80B-T0X-R CPU/CUDA router-portability report

Date: 2026-08-13

## Verdict

`cross_backend_negative`, independently verified from the stored tensors. This is a useful portability counterexample, not a throughput or full-model pass.

The pinned CPU backend and the RTX PRO 2000 Blackwell CUDA backend did not reproduce the same official Qwen3-Coder-Next layer-0 router result on all 16 archived real hidden-state rows:

- ordered top-10 IDs equal: 12/16;
- unordered top-10 expert sets equal: 14/16;
- native-BF16 selected weights bitwise equal: 3/16;
- CUDA call 1 versus CUDA call 2: logits, FP32 probabilities, IDs and native-BF16 weights all bitwise equal;
- all stored numerical tensors finite;
- peak CUDA allocation: 35,850,240 bytes;
- no bank build and no host registration.

## Mechanism

The official router uses a native-BF16 linear projection, FP32 softmax/top-k/normalization, and casts selected weights back to BF16. CPU and CUDA BF16 GEMV accumulation are deterministic within each tested backend but are not bitwise identical across these backends:

- maximum absolute logit difference: 0.0625;
- mean absolute logit difference: 0.0036153793;
- 7,384/8,192 native-BF16 logit values were bitwise equal;
- maximum absolute selected-weight difference: 0.0029296875.

Rows 4 and 5 had an exact CPU rank-10/rank-11 tie and changed the selected expert set:

- row 4: CPU selected expert 458; CUDA selected 214;
- row 5: CPU selected expert 243; CUDA selected 182.

Rows 0 and 12 changed ordered ID sequence without changing the set. The earlier R6-D CPU diagnostic had five zero-margin rows (2, 3, 4, 5 and 10), but only rows 4 and 5 crossed the selected-set boundary on CUDA. This demonstrates that a pinned-backend route claim can be reproducible while a cross-backend route-invariance claim is false.

## Protocol history

The original T0X attempt stopped before CUDA because its runner expected `[1,16,2048]` while the immutable R6-D artifact stores `[16,2048]`. It created no result. T0X-R changed only that shape contract, used create-new paths and then executed once. The invalid pre-execution attempt remains archived.

The independent verifier passed 6/6 replay checks over the stored raw CPU/CUDA tensors. It did not rerun CUDA.

## Consequence

Physical expert execution must consume the official route IDs frozen by the chosen reference backend; recomputing the router on another backend cannot be called bitwise equivalent. T0-R7 therefore tests only clean-process reproducibility of the pinned official CPU backend. It cannot restore the failed R4 cross-backend claim.

## Claim boundary

This result covers 16 real layer-0 hidden-state rows, one official checkpoint revision, the pinned CPU software path and one RTX PRO 2000 Blackwell CUDA stack. It proves neither all prompts, other layers, other devices, model quality, full-depth correctness, throughput nor an industrial breakthrough.
