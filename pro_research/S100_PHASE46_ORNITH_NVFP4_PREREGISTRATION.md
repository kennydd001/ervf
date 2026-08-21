# S100 Phase46 — Ornith-1.5 NVFP4 transfer probe

## Question

Can the exact NVFP4 representation and the route-aware host/GPU split used by
the Nemotron S100 runtime be transferred to Ornith-1.5-35B-A3B before a full
Qwen3.5 hybrid decoder is written?

## Frozen inputs

- Official target: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`
- Abliterated target plus draft:
  `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash`
- GPU: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU (SM120, 8 GiB)
- Native operation: Torch 2.12/CUDA 13.2 `scaled_mm`, corrected C3A-v2
  `SWIZZLE_32_4_4` scale layout.

## Gates

1. The checkpoint declares Qwen3.5-MoE with 40 layers, hidden 2048, 256
   experts, top-8 routing, routed and shared intermediate width 512, and full
   attention every fourth layer.
2. Every layer has one router, 256 complete routed SwiGLU experts
   (`gate_proj`, `up_proj`, `down_proj`) and one complete shared SwiGLU expert.
3. Every selected NVFP4 triple is checkpoint-native U8 packed E2M1 weights,
   E4M3 block scales with group size 16, and an F32 global scale.
4. A known-value two-level NVFP4 smoke executes exactly on SM120.
5. Real routed-gate, routed-down, shared-gate and lm-head tensors execute for
   M1 and M8 and remain finite.
6. Against an independent byte-level decode, M1 normalized RMSE is <= 0.020,
   cosine is >= 0.9990 and normalized max error is <= 0.050.
7. Eight identical M8 rows agree within normalized max difference <= 0.005.

## Claim boundary

Passing Phase46 proves checkpoint representation, shape parametrization and
native matrix-kernel transfer only. It does not prove Qwen3.5 linear-attention
state evolution, router semantics, full-token exactness, DFlash acceptance or
end-to-end token throughput. Those require a separate Ornith decoder adapter.

