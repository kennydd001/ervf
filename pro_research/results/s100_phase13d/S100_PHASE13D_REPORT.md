# S100 Phase 13D — native BF16 block ceiling

Date: 2026-08-18

This component test uses the exact Phase‑12C checkpoint and all BF16 Mamba
input/output matrices, with real-weight rotation above four times the reported
L2 size. For B in `{2,4,8}`, native PyTorch BF16 matrix multiplication is
compared with the current custom row-wise BF16 runtime kernel.

The result is a ceiling test only. Native BF16 uses a different reduction path
and BF16 activations; its output is compared against FP32 accumulation using the
same BF16 weights and inputs. No full model, router, heldout fidelity or
causal-token test is promoted by this artifact. The independent verifier must
remain green and the promotion flag remains false until those gates exist.

Measured median speedup across the twelve resident matrices was 3.26x at B=2,
5.47x at B=4, and 12.07x at B=8. The B=4 component speed gate therefore
passes. Mean row-argmax agreement against the FP32-accumulation reference was
0.979 at B=4; the largest per-case output NRMSE remained below 0.003. These
numbers establish a promising native block ceiling, not a model-level result.
