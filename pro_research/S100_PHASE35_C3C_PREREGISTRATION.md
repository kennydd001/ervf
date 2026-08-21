# S100 Phase 35 / C3C preregistration — fused static NVFP4-A quantizer

Frozen after independently verified C3B.

## Parent

C3B selected `static_1p10`: all real-activation correctness gates green,
32/32 held-out LM-head top-1 retained, and native M8/M1 <=1.0 on all four
families within measurement precision.

## Candidate

One preallocated CUDA kernel quantizes FP32 `[M,K]` activations to packed E2M1
codes and TorchAO row-block-major E4M3 scale layout. The per-family tensor scale
is frozen from C3B calibration; no runtime global-amax reduction is allowed.

Each CUDA block handles one 16-value activation block. It computes the exact
max, E4M3 RNE scale, E2M1 RNE codes and low-nibble-first packing used by the C3B
reference quantizer, and writes scales directly to the native blocked ABI.

## Gates

- Byte-exact packed codes and blocked scale bytes against the C3B reference for
  all four real activation families and M in {1,2,4,8}.
- Native outputs from fused-A must equal native outputs from reference-A.
- No allocations or host synchronization in the timed call.
- Quantizer p50 <=0.10 ms at M8 on all four K shapes.
- Quantizer + native GEMM p50 must be lower than C3B's allocation-heavy
  reference combined path on all four families.
- Warm CUDA-graph capture and two replays must recompute correct output.

C3C is still a component gate. Only a later full verifier trajectory and
end-to-end A/B may make a speed or quality claim.
