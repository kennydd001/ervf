# S100 phase 6 v2 — confirmatory sparsity frontier and exact WAVE downflow

Date frozen: 2026-08-17.

## Quality frontier

Fixed candidates before heldout:

- `alpha_0003`: QFAST, K6, relative ReLU2 cutoff 0.0003;
- budget 1: layer 40 -> K5;
- budget 2: layers 40 and 34 -> K5;
- budget 3: layers 40, 34 and 49 -> K5;
- alpha 0.0003 combined with each nested budget.

All candidates run on validation. Alpha 0.0003 always proceeds to heldout because phase 5 had already shown official validation pass. The largest strict-pass K budget and largest combined arm with official validation pass plus p95 KL <=0.075 proceed to heldout. Timing opens only for heldout official pass.

## Exact WAVE kernel

The existing QFAST/V18 route performs one sparse host-code gather and one masked-down kernel per routed expert slot. WAVE uses a dedicated mirror for every slot and groups slots as W2, W3 or W6.

For each wave:

1. one 2-D CUDA gather stages all route slots in that wave;
2. one 3-D CUDA masked-down kernel computes all slots;
3. per-slot partials remain separate;
4. existing route-ordered reduction and accumulation remain unchanged.

The following gather wave is enqueued on the gather stream while the current wave computes. Expert order, masks, scale bytes, FP4 code bytes and each expert's FMA order are unchanged.

## Gates

Preflight:

- W3 smoke trajectory equals current QFAST;
- finite logits;
- sabotage route must diverge.

Fresh timing:

- independent `BASE_A`, `CAND_A`, `CAND_B`, `BASE_B` processes;
- full arms >=765 samples;
- base/candidate process drift <=1.0 ms;
- VRAM <=7.8 GiB;
- candidate A/B deterministic;
- W2/W3/W6 candidate-vs-base token parity.

S100-single remains <=10.000 ms/useful token plus exact QFAST inheritance or heldout V18-fidelity candidate. Component rates and projections do not count.