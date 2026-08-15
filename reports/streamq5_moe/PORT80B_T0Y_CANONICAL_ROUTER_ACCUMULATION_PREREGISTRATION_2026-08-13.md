# PORT80B-T0Y canonical router accumulation preregistration

Date: 2026-08-13

## Hypothesis

The official BF16 router is backend-dependent because GEMV reduction order differs. A router that fixes every BF16×BF16 product and FP32 addition to round-to-nearest-even, accumulates hidden dimensions strictly in index order, and resolves equal logits by lower expert ID will produce bitwise-identical FP32 logits and ordered top-10 IDs on CPU and CUDA.

This is an invented exact-routing mechanism test on the 16 real R6-D layer-0 hidden states. It does not replace the official router, prove quality, or pass any T0 reference/physical/full-model gate.

## Frozen inputs and algorithm

- R6-D raw SHA-256 `42b9eb25748ce0722f7b3f7c5612069081314eae51b8741a18c39b17abcbdb72`.
- Official shard SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- Inputs `[16,2048]` BF16; router weights `[512,2048]` BF16.
- For each row/expert, start FP32 zero. For `k=0..2047`, compute an IEEE FP32 RN product then a separate IEEE FP32 RN addition. FMA contraction is forbidden.
- Rank logits descending; exact equality is resolved by ascending expert ID. Select 10.
- CPU uses separate NumPy FP32 multiply and add arrays at every k. CUDA uses `__fmul_rn` and `__fadd_rn`, one thread per row/expert, then one deterministic selector thread per row.

## Frozen gates

Primary PASS requires:

1. all 8,192 CPU/CUDA FP32 logits bitwise equal;
2. all 160 ordered top-10 IDs equal;
3. two CUDA calls bitwise equal;
4. all logits finite and IDs unique/in bounds;
5. GPU peak allocation below 256 MiB.

Official CPU and CUDA agreement with the canonical IDs is descriptive only and cannot alter the verdict. No retune/retry is allowed after outputs open. A source/lock preflight is mandatory.

## Claim boundary

PASS would prove a deterministic canonical routing primitive for these shapes/inputs on this CPU/CUDA pair. It would not prove acceptable model quality, routing-weight equivalence, performance, other devices, other layers, or an industrial breakthrough.
