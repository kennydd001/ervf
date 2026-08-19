# S100 Phase 14N2 — native NVFP4 handoff

Historical premises already established on `pro-s100-nativefp4-c2b`:
- checkpoint NVFP4 is packed group-16;
- a lossless Blackwell scale-layout repack was researched;
- native SM120 `scaled_mm` works with BlockWise1x16 and SWIZZLE_32_4_4;
- C2D rotates the four format-preserving checkpoint shapes beyond 4x L2 and
  tests M=1/2/4/8/16.

This pack does not silently turn those premises into a real-weight claim.
It replays the authoritative branch and emits:

- LOSSLESS_REPACK_EVIDENCE_GREEN
- NATIVE_FP4_FREE_M_GREEN
- NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN

If repack evidence cannot be located, the flag is null/incomplete, not false.
