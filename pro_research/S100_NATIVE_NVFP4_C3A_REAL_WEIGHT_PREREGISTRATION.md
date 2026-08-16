# S100 native NVFP4 C3A — real checkpoint weight preregistration

Date frozen: 2026-08-16
Branch: `pro-s100-nativefp4-c2b`
Parent experiment: C2d native FP4 M-scaling.

## Question

Can the Lightning checkpoint's **real packed NVFP4 weight bytes**, real E4M3 group-16 scales and real F32 `weight_scale_2` be consumed by PyTorch 2.12.1+cu132 `F.scaled_mm` on SM120 without changing weight values, while preserving the C2d observation that M=8 is close to the cost of M=1?

This is a representation/geometry experiment. It is not a full-model quality test and it does not claim end-to-end token/s.

## Frozen checkpoint semantics

C0B/C1 established:

- weight payload: two E2M1 FP4 values packed per byte;
- logical block size: 16 values along K;
- local scale: one E4M3 byte per 16 values, natural shape `[N, K/16]`;
- global scale: scalar F32 `weight_scale_2`;
- dequantization semantics: `w = e2m1(code) * e4m3(block_scale) * f32(weight_scale_2)`;
- C1 `SWIZZLE_32_4_4` is a lossless permutation plus padding.

C3A may permute/pad local scale bytes into the native physical scale layout. It must not requantize or alter checkpoint weight codes, local scale values, or the global scale.

## Frozen DUT ABI

The DUT is PyTorch `torch.nn.functional.scaled_mm` in the isolated `.venv-fp4-c2b` environment:

- `torch==2.12.1`, CUDA 13.2;
- packed input dtype `torch.float4_e2m1fn_x2`;
- two scaling levels on both operands:
  - `[BlockWise1x16, TensorWise]`;
  - `[SWIZZLE_32_4_4, NO_SWIZZLE]`;
- output dtype BF16;
- `use_fast_accum=False`.

A is deliberately exact and synthetic: all E2M1 +1 codes (`0x22` packed), local scale +1, global scale +1. This isolates **real B/checkpoint representation** from activation-quantization quality. Activation quantization is deferred to C3B.

## Representative families

Selection is deterministic from safetensors metadata, with exact `lm_head` preferred and shape/name constrained choices for:

1. `lm_head`: N=131072, K=2688;
2. shared expert up projection: N=3712, K=2688;
3. shared expert down projection: N=2688, K=3712;
4. routed expert up projection: N=1856, K=2688.

A selected tensor is valid only if:

- weight shape is `[N, K/2]`;
- scale shape is `[N, K/16]`;
- local scale dtype is F8_E4M3;
- `weight_scale_2` exists and is scalar F32.

## Independent numerical reference

For deterministic sampled output rows, C3A decodes checkpoint bytes with stdlib code independent of the Torch dtype views used by the DUT:

- low nibble = even K coordinate, high nibble = odd K coordinate;
- explicit E2M1 lookup table;
- explicit E4M3 decoder;
- F32 global scale via `struct.unpack`;
- `math.fsum` over K.

Because A is exactly +1, reference output row n is the sum of the dequantized B row.

The independent verifier re-reads the checkpoint and recomputes these reference samples rather than importing the diagnostic implementation.

## Correctness gates

All are frozen before result inspection:

- C3A_G1: Torch 2.12.1+cu132, CUDA available, SM120+, public `F.scaled_mm`, FP4 dtype and required enums present.
- C3A_G2: C1 parent result is `repack_lossless` or `repack_lossless_high_padding` and its correctness gates are green.
- C3A_G3: all four representative checkpoint triples satisfy the frozen packed/group-16 metadata contract.
- C3A_G4: a synthetic **two-level** scale smoke test executes and matches its known value.
- C3A_G5: real-weight native M=1 and M=8 calls execute and produce finite BF16 outputs for all four families.
- C3A_G6: for every family, sampled M=1 normalized RMSE <= 0.020 and cosine >= 0.9990 versus the independent dequantized reference.
- C3A_G7: for every family, sampled normalized max-absolute error <= 0.050 versus reference max magnitude.
- C3A_G8: for every family, all eight identical M=8 rows agree with the M=1 row with normalized max difference <= 0.005.
- C3A_G9: checkpoint weight/local-scale/global-scale SHA256 values recorded by the diagnostic match a fresh read in the independent verifier.

A correctness failure closes the direct real-checkpoint representation route until the mismatch is explained. It must not be relabeled as a model-quality failure.

## Cold timing protocol and performance gate

C2d capped rotation at 24 matrices, which was insufficient to reach 4x L2 on the smaller shapes. C3A fixes that protocol.

For each family, the timing arm creates enough distinct GPU copies of the real B payload + physical local-scale buffer that:

`rotation_working_set_bytes / L2_bytes >= 4.0`

There is **no fixed 24-copy cap**. If memory cannot satisfy this condition, that family's cold timing arm is `not_run_memory_gate`; it may not be counted as a performance pass.

Frozen performance gate:

- C3A_P1: every successfully timed family records working-set/L2 >= 4.0;
- C3A_P2: M8/M1 p50 <= 1.15 for at least 3 of the 4 representative real-weight families;
- C3A_P3: `lm_head` M8/M1 <= 1.15 if its timing arm runs.

Performance is secondary to correctness. A performance miss does not invalidate real-weight representational correctness.

## Claim boundary

A green C3A proves only:

1. the real Lightning checkpoint NVFP4 B representation can feed the native SM120 two-level scaled-mm path after lossless scale swizzle;
2. native outputs agree with an independent dequantized reference under exact +1 activations;
3. real-weight M geometry is measured under an honestly >=4x-L2 rotation protocol.

It does **not** prove real activation quantization quality, MoE grouped-kernel support, stateful/Mamba integration, full-model continuation parity, or 100 tok/s end-to-end. Those require C3B/C3C.
