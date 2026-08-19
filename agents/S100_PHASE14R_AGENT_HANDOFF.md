# S100 Phase 14R — repair handoff

## Confirmed, do not rerun

The completed DFlash2 run is a valid closure for the current runtime:

- perfect-draft verifier ceiling remains approximately 56 tok/s;
- no resident 4K drafter envelope;
- suffix correction showed no useful validation signal;
- the selector captured about 1.1% of its oracle headroom;
- training remains closed.

## Why 14D is still unresolved

The published Phase-14 summary has `14D_complete=false`. The extended component
test was executed inside the fully resident quality parent, which reported zero
free VRAM, and then created both a cloned BF16 weight and a contiguous
transposed copy. Large Mamba native matmuls fell to approximately 11 ms, versus
approximately 0.2 ms in the earlier lean Phase-13D run. This is consistent with
WDDM paging/oversubscription, not a valid native Tensor-Core negative.

Phase 14R uses a lean runtime and aliases the existing weight allocation. No
second weight copy exists.

## Why 14B2/14E2 remain unresolved

14B2 failed because the installed eager MoE implementation returned `None`
while `runtime.step()` unpacks `(ids, weights)`. The repair normalizes that
return value to `(None, None)` without changing the computation.

14E2 never ran because its capture dependency was missing.

## Final flags

- `NATIVE_BLOCK_RUNTIME_BUILD_OPEN`
- `SUBSPACE_RUNTIME_BUILD_OPEN`
- `EXPERT_BASIS_RUNTIME_BUILD_OPEN`

A technical or paging-contaminated result remains `null`, never `false`.
