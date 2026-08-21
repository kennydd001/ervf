# S100 Phase58 — Ornith direct-L2 FP8 H4 projections

## Question

Does the adopted Nemotron direct-L2 H4 mechanism transfer to Ornith's real
FP8 E4M3 attention projections when the BF16 weight decoder is replaced by an
exact E4M3 decoder?

## Frozen artifacts

- Pottokao revision
  `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash@d60d98b0b2feeabca19196005f4ac378279e2f25`.
- Official revision
  `ornith-ai/Ornith-1.5-35B-A3B-NVFP4@0f0b1b59b879ccde1353e6ebd0fb10c204d4c544`.
- Real layer-20 linear-attention QKV, Z and output projections from both
  checkpoints.
- Real Pottokao layer-23 full-attention Q, K, V and output projections.
- RTX PRO 2000 Blackwell Laptop GPU, native CUDA execution through CuPy.

## Arms

1. `m1_x4`: four production-order single-row kernels, one launch per row.
2. `m4_direct_l2`: one weight pass and four independent accumulators, with the
   four small FP8 activation rows read through L2 and no dynamic staging.

Both arms decode identical FP8 bytes, use the same scalar input/weight scales,
execute each output's `k` loop and two-level reduction in the same order, and
write FP32 outputs. Timing uses GPU events after warm-up with weights resident
on device.

## Gates

1. The CUDA E4M3 decoder matches PyTorch for all 254 finite byte patterns; both
   NaN encodings are classified as NaN.
2. M4 output is bitwise identical to four M1 outputs for every real matrix.
3. Every output is finite and repeat execution is bitwise deterministic.
4. Both kernels compile with zero local memory; M4 uses at most 64 registers.
5. Every matrix has M4/M1x4 speedup greater than 1 and median speedup is at
   least 2.
6. Official/Pottokao latency ratios for matching layer-20 shapes stay within
   0.8–1.25.

## Claim boundary

This phase ports and times real FP8 projection GEMVs only. It does not include
Qwen3.5 gated-delta recurrence, convolution, RoPE/softmax attention, MoE,
router, residual, DFlash acceptance, sampling, or end-to-end tok/s.
