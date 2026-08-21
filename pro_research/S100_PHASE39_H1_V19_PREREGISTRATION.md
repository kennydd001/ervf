# S100 Phase39 — context-1024 H1 V19 carry-over

## Question

Claude's Phase31C adjudication reports the fastest exact single-token path as
the V6 H1 graph at roughly 21.95 ms/token. The repository already contains an
older, fully adjudicated V19 stack (resident down-scale planes, overlapped
gather/down execution, and the exact block-per-(h,p) SSM kernel) that measured
19.74 ms/token on a shorter-context harness. This experiment asks whether that
stack carries over, bit-exactly, to the canonical Phase31 context-1024 H1
window.

This is not a speculative-decoding experiment. It measures one target-model
token per CUDA-graph replay and one synchronous token readback.

## Frozen arms

- `BASE_A`: V6 device-row MoE, context 1024, fresh process.
- `V19`: V18 combined MoE plus the V19 exact SSM replacement, context 1024,
  fresh process.
- `BASE_B`: V6 device-row MoE, context 1024, fresh process after `V19`.

All arms use the same official Nemotron-3.5-Lightning NVFP4 snapshot, canonical
Phase20B token trace, FP32 KV cache, Phase22 graph-safe attention, 16 warmup
tokens, and 128 measured teacher-forced tokens.

## Frozen gates

- `G39-C1`: every generated token in all three arms equals the canonical next
  token.
- `G39-D1`: `abs(BASE_A median - BASE_B median) <= 1.0 ms`.
- `G39-P1`: `V19 median <= 0.95 * baseline midpoint` (at least 5% faster).
- `G39-P2`: V19 must fit at cache capacity 72 without changing capacity or
  evicting an expert-cache allocation.

If C1 or P2 fails, the candidate is rejected. If D1 fails, the measurement is
unstable and no speed claim is made. If C1, D1, P1, and P2 pass, V19 becomes
the exact H1 anchor. It does not supersede a faster exact H4 verifier.

