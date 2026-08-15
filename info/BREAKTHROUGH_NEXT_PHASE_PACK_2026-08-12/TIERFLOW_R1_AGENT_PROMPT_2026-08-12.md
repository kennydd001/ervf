# Agent prompt — TierFlow-R1

Open a training project independent from all frozen post-hoc inference work.

## Model

Start with a reproducible 100M–300M decoder MoE:

- 12–18 layers;
- 16 or 32 experts;
- top-2/top-4;
- fixed training-token and compute budget.

## Variants

1. standard MoE;
2. StickyMoE;
3. ReMoE-style router fine-tuning;
4. closest reproducible TriRoute controller;
5. TierFlow route-only;
6. TierFlow route + progressive full-rank pages.

## TierFlow

Persistent route state:

```text
R_t = Update(R_t-1, delta_R_t)
|delta_R_t| <= r
```

Progressive weight pages:

```text
W = W2 + D3 + D4 + D5
```

Controller inputs:

- hidden state;
- cache residency;
- transfer queue;
- DMA slack;
- route edit history;
- router margin;
- uncertainty;
- bit budget.

Loss:

```text
LM loss
+ mean critical-path latency
+ CVaR95 latency
+ route-edit penalty
+ tier dropout
```

## Gates

- quality regression <=1%;
- critical expert bytes reduced >=4x;
- worst-case new expert loads reduced >=8x;
- measured p95 improved >=2x;
- no p99 collapse;
- works under a second memory hierarchy.

Do not scale to billions until every gate passes.
