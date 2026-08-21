# S100 Phase79 — learned DFlash-to-target route projection

## Question

Can a small projection learned from earlier aligned speculative events convert
DFlash final hidden state into useful target-router logits?

## Frozen split and alignment

- Same Phase78 trace and prefix alignment.
- Events 0..4 are calibration only; events 5..8 are held-out chronological
  evaluation. No online updates occur during evaluation.
- Same-position and next-position (`+1`) alignments are both retained; `+1` is
  primary.

## Frozen correctors

For each of 40 target routers, start from `router(DFlash result_norm)` and learn
its 256-logit residual against the exact target router logits:

- mean residual bias;
- cosine k-nearest residual mean, k=4;
- normalized linear-kernel ridge residual with lambda 0.1 and 1.0.

Each event contributes only its authoritative target-length prefix. Candidate
sets are stable unions of per-position top-8, bounded to 32 per layer over the
first four evaluable positions.

## Gates

1. Phase78 callback alignment and Phase76 exact-router contracts are green.
2. A `+1` arm reaches at least 80% assignment recall or 95% H4 unique-route
   recall on events 5..8.

Passing would authorize multi-prompt training and cache-aware runtime timing;
failure closes this single-trace low-data DFlash projection.
