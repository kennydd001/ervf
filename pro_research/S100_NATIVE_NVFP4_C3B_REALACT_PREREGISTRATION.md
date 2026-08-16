# S100 Native NVFP4 C3B — real-activation preregistration

Date: 2026-08-16
Parent: C3A-v2 corrected TorchAO row-block-major scale layout
Target: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on SM120 / torch 2.12.1+cu132

## Claim boundary

C3B asks one question only: does the native SM120 NVFP4 `scaled_mm` geometry that
survived real checkpoint weights also survive **real causal Lightning activations**
when A is quantized to NVFP4 semantics?

This is NOT an end-to-end tok/s claim, NOT an integrated decoder, and NOT a claim
that the reference PyTorch quantizer is fast enough for production. The
quantizer in this phase is intentionally transparent and allocation-heavy so its
numerics are auditable. Its timing is recorded only as an upper bound; a fused
C3C quantizer is required before integration.

The activation capture uses the V6 arithmetic stack (selective ERVF + batched
MoE) in eager submission. V13 H-SCALE and V14G/B3, which form V18, change only
byte placement / transfer timing and were separately proven bit-exact. Therefore
the captured values are arithmetic-equivalent to the V18 record stack, while
being much easier to instrument without perturbing a captured CUDA graph.

## Frozen data split

Capture 64 consecutive causal decode steps from the first registered anchor
prompt.

- rows 0..31: calibration only
- rows 32..63: held-out quality and timing input
- target MoE layer: first MoE layer in the runtime pattern
- routed-up representative: expert 0 of that same layer

Captured values per step:

- target-layer `normed` (input to shared_up and routed_up)
- exact shared-up post-ReLU2 activation (input to shared_down)
- final `normed` (input to lm_head)
- exact ERVF token id
- exact ERVF top-5 ids/logits

## A quantization semantics

Block size K=16. FP4 E2M1 codes use round-to-nearest-even, finite saturation.
For each 16-value block:

1. `block_scale = amax(block) / 6`
2. `scaled_block_scale = clamp(block_scale / tensor_scale, 2^-6, 448)`
3. cast the block scale to FP8 E4M3
4. divide A by `(tensor_scale * quantized_block_scale)`
5. clamp to [-6, 6], round to FP4 E2M1, pack low nibble first
6. swizzle the FP8 scale bytes in the same TorchAO `to_blocked` row-block-major
   128x4 outer-tile order validated by C3A-v2

Two arms are frozen before measurement:

- DYNAMIC: `tensor_scale = amax(current MxK) / (448*6)`
- STATIC_1P10: per-family tensor scale from the 32 calibration rows,
  `1.10 * calibration_amax / (448*6)`. The 10% margin is frozen to protect the
  held-out set from mild range growth; it may not be retuned after seeing C3B.

If both arms pass, STATIC_1P10 is preferred for C3C because it removes the
per-GEMM global-amax reduction. If STATIC fails and DYNAMIC passes, DYNAMIC is
the C3C parent. No post-hoc third scale arm is allowed in C3B.

## Shapes and M values

- lm_head: real Lightning lm_head triple, K=2688
- shared_up: target layer shared_experts.up_proj, K=2688
- shared_down: target layer shared_experts.down_proj, K=3712
- routed_up: target layer experts.0.up_proj, K=2688

M values: 1, 2, 4, 8. Rows are distinct consecutive held-out decode
activations; duplicating one row eight times is forbidden.

## Frozen correctness gates

All gates are fail-closed.

- C3B_G1: C3A-v2 layout preflight PASS and C3A real-weight representation is
  correctness-green.
- C3B_G2: every capture binary matches the SHA256 in the capture manifest.
- C3B_G3: FP4 nibble + scale-layout + native known-value preflight is exact.
- C3B_G4: every native output used by C3B is finite.
- C3B_G5_DYNAMIC_LOCAL: for all four families, held-out M=8 sampled-output
  NRMSE <= 0.080, cosine >= 0.9950, normalized max absolute error <= 0.200.
- C3B_G6_STATIC_LOCAL: same thresholds for STATIC_1P10.
- C3B_G7_DYNAMIC_LM: over all 32 held-out lm_head rows, native top-1 equals
  exact ERVF top-1 in >= 90% of rows AND native top-1 lies inside exact ERVF
  top-5 in >= 97% of rows.
- C3B_G8_STATIC_LM: same thresholds for STATIC_1P10.

The reference for local matrix metrics is the original FP32 activation times an
independently dequantized checkpoint-B sample (FP4 codes * E4M3 block scales *
checkpoint global scale), accumulated on CPU in FP64.

## Frozen performance-geometry gates

These gates use PREQUANTIZED A so they answer only whether real A values disturb
the native M geometry. B is rotated over >=4x L2 whenever memory permits, exactly
as in C3A. Reference-quantizer and combined timings are recorded but cannot be
promoted to a production claim.

- C3B_P1: every measured cold set >= 4.0x L2.
- C3B_P2_DYNAMIC: M8 wall-time / M1 wall-time <= 1.20 on at least 3/4 families,
  and lm_head <= 1.20 if measured.
- C3B_P3_STATIC: same for STATIC_1P10.

C3B becomes `real_activation_native_candidate` if parent/capture/preflight are
green and at least one arm passes both its correctness gates and its performance
geometry gates.

## What opens next

A green C3B opens C3C: a fused, preallocated SM120 activation quantizer with the
selected tensor-scale policy, timed in the actual hot path. Only after C3C may a
native A-quantization cost be used in an integration budget. C4 then replaces
selected real decoder GEMVs and runs end-to-end A/B; C5 is the causal M-way / expert
union path required to amortize real weight traffic per accepted token.
