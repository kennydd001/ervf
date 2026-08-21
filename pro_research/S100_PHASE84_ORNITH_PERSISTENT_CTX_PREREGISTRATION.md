# S100 Phase84 — persistent target-only context verifier

## Frozen question

What is the real wall-clock cost of the final fully authoritative target H4 at
ctx64 when the integrated verifier uses persistent dense weights, real
attention state/KV and a physical LRU52 expert cache without DFlash? Longer
tokenizer-repeated contexts are separate synthetic target-only stress tests;
they cannot be relabeled as authoritative generated target sequences.

## Runtime contract

- Up to ctx64, token IDs come directly from the committed authoritative
  target/reference trace. Above ctx64, only that prefix is authoritative; the
  suffix repeats the prompt through the checkpoint tokenizer and is explicitly
  labeled synthetic stress input.
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
  every persistent recurrent/KV-state hash, final-normalized bit and ERVF ID
  from the measured run.
- Synchronized component boundaries may be collected in that validation repeat
  to localize discrepancies. They are diagnostics and never replace or enter
  the unsynchronized primary wall-clock result.

No calibrated waits, component sums, oracle prefetch, captured future routes or
DFlash signals are permitted.

## Gates

1. The declared sequence authority and authoritative prefix are exact.
2. Physical page/cache invariants hold after every layer and D2D promotion bytes
   remain zero in the runtime copy ledger.
3. Every output/state is finite and a fresh rerun is deterministic in routes,
   final norm and ERVF IDs.
4. Same-input router IDs and ERVF top-1 are exact against their independent
   control calculations.
5. Report complete wall-clock ms/H4 without converting it to output tok/s. Only
   ctx64 may carry the fully-authoritative trace label under the current input.

Transport optimization remains blocked until the strict ctx64 gate passes with
observed zero D2D and bit-exact persistent-state hashes.
