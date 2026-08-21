# S100 Phase78 — DFlash hidden-to-target-route signal

## Question

Does the matched Pottokao DFlash decoder's real `result_norm` carry enough
information to shortlist target MoE routes before target verification?

## Frozen alignment

- One fresh deterministic llama.cpp speculative run, target and DFlash in the
  same process, K=7, target ubatch 256, greedy seed 7801.
- Callback instrumentation is preserved as
  `pro_research/patches/llama_ornith_speculative_trace.patch`, applied to
  llama.cpp commit `9558fa44c92746a58dd07ad1bf0c889715b938a6`.
- The explicit nine `target_batches` must match the lengths of the final nine
  40-layer target callback groups. Earlier shape-2 callback groups are
  initialization and excluded.
- Technical amendment after the first callback audit: DFlash always materializes
  its trained H8 block, while the target truncates the final verification batch
  to H7/H4 at the generation budget. Therefore each aligned DFlash tensor must
  have at least the target length and only that authoritative prefix is scored.
- Token alignment arms are frozen as same position and `+1`: DFlash position
  `j` predicts target input position `j+1`. The latter follows next-token hidden
  semantics and is primary.

## Frozen route probes

- Apply each exact BF16 target router directly to DFlash `result_norm`.
- Measure per-position top-8 assignment recall.
- Build stable H4 candidate unions from the first four aligned positions with
  budgets 32 and 64. Budget 64 is diagnostic because H4 itself has at most 32
  unique authoritative experts.

## Gates

1. Target event lengths equal batch metadata and every DFlash event covers that
   target prefix.
2. The same target router/checkpoint contract from Phase76 remains exact.
3. Primary `+1` top-8 assignment recall is at least 80%, or its H4 budget-32
   unique-route recall is at least 95%.

A pass authorizes a cache-aware DFlash route-prefetch experiment. A failure
means raw DFlash final hidden is not directly interchangeable with target
intermediate router inputs; a learned projection remains separate.
