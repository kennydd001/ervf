# S100 Phase 14R — repair the still-open survivors

Date: 2026-08-19

DFlash2 is frozen closed for the current runtime and is not rerun.

14D was not scientifically closed: it ran in a zero-free-VRAM parent and
allocated cloned/transposed BF16 weights, producing paging-like 11 ms native
matmuls. Phase 14R uses a lean runtime and DLPack weight aliases with no copy.

14B2 failed only because eager runtime.step() unpacked a graph-oriented MoE
wrapper that returned None. Phase 14R normalizes the return to `(None, None)`.

14E2 runs only after the repaired real MoE captures exist.

Final flags are tri-state:
- NATIVE_BLOCK_RUNTIME_BUILD_OPEN
- SUBSPACE_RUNTIME_BUILD_OPEN
- EXPERT_BASIS_RUNTIME_BUILD_OPEN
