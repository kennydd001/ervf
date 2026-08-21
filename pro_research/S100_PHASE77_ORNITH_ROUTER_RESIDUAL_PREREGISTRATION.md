# S100 Phase77 — online future-router residual correction

## Question

Can the Phase76 future-router proxy be corrected using the observed relation
between earlier source activations and destination-router residuals, while
remaining strictly chronological?

## Frozen setup

- Same 64-token hidden trace, exact safetensors routers, LRU52 replay and
  token-32 test boundary as Phase76.
- Leads 1 and 2, total candidate budget 32 per destination layer/H4.
- Base signal is Phase76 `direct`: destination router on the source layer's
  normalized activation.
- At every test block, correction state contains only tokens before that block.

## Frozen correctors

- last observed 256-logit residual;
- exponentially weighted residual mean with decay 0.9;
- cosine-nearest residual means with k = 1, 4 and 8 source activations;
- linear-kernel ridge residual regression with lambda = 0.01, 0.1 and 1.0,
  scaled by the training-kernel diagonal mean.

The correction predicts `exact_destination_logits - proxy_logits`. Layers before
the lead retain Phase76's chronological frequency fallback.

## Gates

1. Phase76 exact-router and trace contracts remain green.
2. A lead-2 corrector covers at least 95% of unique real LRU52 misses.
3. Its optimistic projected latency is at or below 4000/65 ms/H4.

Failure closes low-data online residual correction on this trace; it does not
close a trained cross-prompt auxiliary route head.
