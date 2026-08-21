# S100 Phase74 — Ornith causal route-prefetch audit

## Question

Can an online predictor that has seen only earlier target routes cover enough of
the next H4 block's real LRU52 misses to replace the perfect-future-route oracle
used in Phase72/73?

## Frozen data and split

- Phase70's bit-repeatable 128-token, 40-layer llama.cpp route trace.
- Blocks beginning before token 32 are history/warm-up only.
- Blocks beginning at tokens 32..124 are evaluated chronologically. Predictor
  state may include earlier evaluated tokens, never the current/future block.
- Cache capacity is 52 experts per layer and authoritative target routes alone
  update persistent cache metadata.

## Frozen predictors

- perfect current-H4 oracle (ceiling/control);
- previous H4 rows;
- most-recent unique experts, cumulative frequency and first-order route
  transition scoring, each at unique budgets 8, 16, 24 and 32;
- transition/frequency hybrid at the same four budgets.

Ranked expert lists are deterministically packed into four top-8-shaped rows so
the production rolling-prefetch controller is exercised unchanged. Prediction
mistakes remain temporary staging entries and cannot change target arithmetic.

## Metrics and gates

The primary metric is recall over actual unique LRU52 miss groups. Secondary
metrics are staged precision, false prefetches, uncovered groups and target
route assignment accuracy. The implementability projection is deliberately
optimistic: Phase69's all-hot floor plus Phase73's selected LRU exposed tail,
plus Phase71's measured serial milliseconds per uncovered expert group.

1. The trace/control contracts pass and the oracle leaves zero uncovered miss
   groups.
2. All evaluated blocks are chronological and contain no current-block input to
   a causal predictor.
3. A causal arm reaches at least 95% unique-miss recall.
4. That arm's optimistic projected floor remains at or below 4000/65 ms/H4.

Failure closes route-history-only lookahead for the 65 tok/s path. It does not
close predictors using DFlash hidden states, target router margins, or a learned
auxiliary head; those require new evidence.
