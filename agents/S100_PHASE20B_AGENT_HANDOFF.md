# Phase 20B handoff

This is the first full-target perfect-draft economics experiment.

Perfect drafts remove drafter uncertainty only. Every block contains 4 useful
tokens from one sequence. Report block_ms, ms/useful_token and target_only_tok_s.
Never convert H-way work into aggregate throughput.

`S100_SINGLE_ACHIEVED` must remain false in this phase even if target-only
verification exceeds 100 useful tok/s, because drafter cost is not included.

Absolute target-only gate: median H4 block <= 40.000 ms at every measured
context. Preferred drafter-headroom gate: <=32.000 ms.

MoE grouped path must preserve sigmoid router, correction bias, top-6 ids,
route normalization/scaling, shared expert, routed ReLU^2 and route-slot
accumulation order per token. Cache scheduling may change; model math may not.
