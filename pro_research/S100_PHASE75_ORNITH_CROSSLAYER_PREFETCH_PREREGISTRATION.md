# S100 Phase75 — causal cross-layer Ornith prefetch

## Question

Can routes already computed inside the current target H4 block predict the
expert misses two or more layers ahead, allowing real copy/compute overlap
without perfect future knowledge?

## Frozen design

- Same Phase70 128-token route trace, LRU52 cache and token-32 chronological
  evaluation boundary as Phase74.
- For destination layer `D` and lead `q`, the predictor sees current-block
  routes only through layer `D-q`, plus complete routes from earlier tokens.
- Online source-expert to destination-expert co-occurrence counts are trained
  exclusively on earlier tokens. Scores sum the learned destination counts for
  the current block's source experts, then break ties by destination frequency,
  recency and expert ID.
- Leads 1, 2 and 4; candidate budgets 8, 16, 24 and 32. Layers before the lead
  use destination-layer historical frequency only.
- A perfect oracle remains the control. Target routes alone commit LRU state.

## Gates

1. Oracle/control and chronological leakage checks pass.
2. A physically useful lead-2-or-greater arm covers at least 95% of actual
   unique LRU52 misses.
3. The same optimistic Phase74 latency projection remains at or below 4000/65
   ms/H4.

Failure closes route-ID association alone. Hidden-state or router-logit
predictors remain separate hypotheses.
