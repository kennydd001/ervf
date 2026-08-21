# S100 Phase84 — persistent target-only context verifier

## Frozen question

What is the real wall-clock cost of the final authoritative target H4 at
ctx1024 when the integrated verifier uses persistent dense weights, real
attention state/KV and a physical LRU52 expert cache without DFlash?

## Runtime contract

- The token sequence is produced by the checkpoint tokenizer from the committed
  target/reference prompt. The first 64 IDs must exactly match the committed
  trace.
- Dense FP8 weights, norms, routers, shared experts and the head remain resident
  on GPU.
- Every layer owns 52 logical expert pages plus one staging physical page.
- A miss is discovered only after the current target router. Its six compressed
  segments are copied mmap -> pinned staging -> the current staging GPU handle.
  Promotion swaps handles; it performs no D2D payload copy.
- Linear-attention state and full-attention KV persist across all preceding H4s.
- The final measured H4 includes embedding, 40 layers, router/cache/real H2D,
  routed/shared SwiGLU, final norm, native shortlist and exact ERVF rerank.
- Wall time is `perf_counter` around the complete host+GPU call with a final
  stream synchronization. CUDA events are diagnostic only.
- Independent CPU route parity and full-head control logits run only in a fresh
  validation repeat outside the primary performance epoch. The repeat still
  starts from empty recurrent/KV/cache state and must reproduce every route,
  final-normalized bit and ERVF ID from the measured run.

No calibrated waits, component sums, oracle prefetch, captured future routes or
DFlash signals are permitted.

## Gates

1. The authoritative first-64 token prefix is exact.
2. Physical page/cache invariants hold after every layer and D2D promotion bytes
   remain zero.
3. Every output/state is finite and a fresh rerun is deterministic in routes,
   final norm and ERVF IDs.
4. Same-input router IDs and ERVF top-1 are exact against their independent
   control calculations.
5. Report the complete ctx1024 wall-clock ms/H4 without converting it to output
   tok/s unless all gates pass.

Longer contexts remain blocked until this ctx1024 gate is measured and the
60.084/74.787 discrepancy is re-evaluated against this continuous epoch.
