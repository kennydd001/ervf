# S100 Phase 16 agent handoff

Phase 15 established that full native-BF16 cumulative substitution is unsafe.
Attention-only is materially closer. Two important routes were not measured:
15C died from repeated pinned-bank allocation and 15D died on the experimental
FP32-output CUBLAS contract.

Phase 16 returns to the exact D2 primitive that actually measured fast:
`torch.mm(x_bf16, W_transposed_contiguous).float()`.

New first-principles hypothesis: the production Mamba SSM step is affine in
the previous state, `s_t = a_t*s_(t-1)+b_t`. Affine pairs compose
associatively, so H token states can be computed by parallel prefix scan once
the token-local x/B/C/dt values are available. 16E validates that algebra on
real production token parameters before any block kernel is built.
