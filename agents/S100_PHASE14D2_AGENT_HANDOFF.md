# S100 Phase 14D2 — agent handoff

Phase 13D/H measured the strongest positive signal so far:
- native BF16 Mamba block execution was several times faster at B=4;
- BF16 attention Q/O was also roughly 6x faster at B=4;
- those were component results, not a full-model quality result.

This pack answers the missing questions:
1. does native BF16 still win at B=1 under cold weight conditions?
2. does the numerical change survive the frozen quality suite?
3. does B=4 remain a large enough hardware ceiling to justify rebuilding the
   perfect-draft verifier around native Tensor Cores?

Do not confuse eager quality timing with production graph timing. The eager run
exists only to make Torch/CUDA native BF16 calls part of the causal target
runtime and to measure fidelity. A later CUDA/CUTLASS graph-integrated runtime
is required before an end-to-end speed claim.
