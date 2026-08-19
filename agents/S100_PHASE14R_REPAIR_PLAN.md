# S100 Phase 14R — repair the still-open survivors

Date: 2026-08-19

## Confirmed closure

DFlash2 training is closed for the current runtime. The complete Phase-14F run measured a perfect-draft verifier ceiling near 56 tok/s, no resident 4K drafter envelope, and no useful suffix-correction or path-selector transfer signal.

## Why the rest of Phase 14 is not closed

The published Phase-14 summary reports `14D_complete=false`, `14B2_complete=false`, and `14E2_complete=false`.

- 14D was measured inside the fully resident quality parent with zero reported free VRAM, then cloned/transposed each BF16 weight into Torch. Several large native GEMMs fell to roughly 11 ms although the earlier lean 13D test measured roughly 0.2 ms. This is a memory-paging/instrumentation result, not a valid native Tensor-Core falsification.
- 14D quality never ran because the auto-selected worktree lacked the frozen Phase-3 full trace.
- 14B2 failed because the installed eager MoE wrapper returned `None` while `runtime.step()` requires `(idx, weights)`.
- 14E2 depended on the missing 14B2 captures.

## Repair tests

1. **14D-ZC component:** use a lean `LightningRuntime`; alias resident CuPy BF16 weights into Torch by DLPack; no clone, no transposed copy; use `torch.nn.functional.linear`; time with CUDA events; log free VRAM and reject any run showing paging-like bandwidth.
2. **14D-ZC quality:** run the current quality parent from a worktree containing `S100_PHASE3_V18_TRACE_FULL.npz/json`; normalize the eager MoE return contract; replace BF16 GEMV numerics with zero-copy native BF16 linear; strict validation then frozen heldout.
3. **14B2 repair:** normalize `_moe` return to `(None,None)` for eager capture and rerun output-aware reduced-rank regression.
4. **14E2 repair:** consume the resulting real MoE input/activation/route captures and run the decoded activation-weighted shared-basis screen.

## Final flags

- `NATIVE_BLOCK_RUNTIME_BUILD_OPEN`
- `SUBSPACE_RUNTIME_BUILD_OPEN`
- `EXPERT_BASIS_RUNTIME_BUILD_OPEN`

DFlash2 flags remain frozen false and are not rerun.

Missing evidence remains `null`; only technically complete gates may become `true` or `false`.
