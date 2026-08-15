# HET-NEXT L0 PH0X-R4 — CUDA compile/staging diagnostic preregistration

Date: 2026-08-13. Diagnostic only; no kernel launch and no numerical/device positive claim.

The immutable PH0X-R3 result SHA-256 is `e5fea8e2609f11dd294733645c9a4ecb08892c9d2070de33baacbd1a74b0df7c`. It proves Intel produced 512/512 words bitwise equal to the CPU oracle (`e8a00c17...`), all counters one, sentinel overwritten, and clean Intel cleanup. NVIDIA failed before compile/launch at Python `memoryview` assignment into pinned host memory.

This diagnostic performs only:

1. rebuild the same official record and verify record SHA `e3b10ab3...`;
2. allocate one 675,840-byte CUDA pinned-host buffer;
3. stage with `ctypes.memmove`, read back with `ctypes.string_at`, and require byte-identical SHA;
4. compile/retrieve (but never launch) the exact R5 candidate CUDA kernel;
5. release the pinned allocation and report identity/compiler outcome.

The CUDA kernel retains the exact width-8 reduction DAG. It replaces cooperative-groups syntax with `lane=threadIdx.x&7`, subgroup `(threadIdx.x>>3)&31`, and `__shfl_down_sync(__activemask(), value, offset, 8)`. All 32 warp lanes participate; constant mask `0xff` is forbidden.

No Intel call, no CUDA kernel launch, no output/counter allocation, no H2D/D2H, and no retry.
