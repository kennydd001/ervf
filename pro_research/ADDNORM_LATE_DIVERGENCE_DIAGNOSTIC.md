# AddNorm late-divergence diagnostic

Frozen before running this diagnostic on target hardware.

Observed PV2-10 facts:

- direct random-input hidden output bitexact;
- direct random-input RMSNorm output bitexact;
- fused microkernel ~1.15x versus separate add+norm;
- graph rollout exact on two prompts but first token divergence at generated
  token 124 on `The history of computing began when`.

The fused arithmetic appears to reproduce the production source, so a late token
mismatch is not enough to identify whether the cause is a rare arithmetic input,
graph scheduling, cache/event timing, or harness state.

## Diagnostic

Run the troublesome prompt through the V6 graph-mode kernels manually, one token
and one layer at a time. At each residual transition, duplicate the exact real
`h` and `acc` tensors:

- reference copy: production `add_inplace` then production `rmsnorm_bf16w`;
- candidate copy: `pv2_add_rmsnorm_bf16w`.

Synchronize only for diagnosis, compare the updated hidden vector and normalized
vector as raw uint32 bits, then continue the actual model state with the
**reference** outputs. Mamba/KV/cache state is updated only once, so the
comparison itself cannot perturb the causal reference path.

Stop on the first bit mismatch and report generated-token index, layer, vector,
first differing element and mismatch count. If no mismatch occurs through at
least 160 generated tokens, the direct arithmetic hypothesis is rejected and the
next investigation must focus on captured-graph scheduling/state rather than
changing the fused formula.

This diagnostic is not timed and cannot produce a speed claim.
