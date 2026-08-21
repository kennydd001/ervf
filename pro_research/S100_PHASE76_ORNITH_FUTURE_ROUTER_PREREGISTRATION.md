# S100 Phase76 — Ornith future-router probe

## Hypothesis

The normalized residual stream changes slowly enough that destination router
`D` can be evaluated early on the real target activation from layer `D-q`.
Unlike route-ID association, this uses each destination router's own weights and
can rank experts that have never appeared in route history.

## Frozen evidence

- A fresh 64-token CPU llama.cpp callback trace from the exact Phase70 prompt
  and GGUF, capturing `attn_post_norm` for all 40 layers plus authoritative
  top-8 routes. Tokens must equal Phase70's first 64 tokens.
- Exact BF16 router and post-attention RMSNorm weights from the Pottokao
  safetensors checkpoint already proven compatible with this GGUF.
- Tokens 0..31 calibrate/history; tokens 32..63 are the chronological test.
- LRU52 state is updated only by authoritative routes.

## Frozen arms

- leads 1, 2 and 4;
- candidate budgets 8, 16, 24 and 32 unique experts per layer/H4;
- `direct`: destination router on source layer's captured normalized activation;
- `normswap`: divide out the source RMSNorm weight and apply the destination
  RMSNorm weight before the destination router;
- each of the above with an online per-destination-expert mean logit correction
  learned only from earlier tokens.

For each token the proxy contributes its top eight; their stable union is
trimmed to the budget by maximum H4 proxy score. Layers before the requested
lead use chronological destination-route frequency. The exact same-layer router
is a parity control; the actual route union is the oracle ceiling.

## Gates

1. Token/tensor contracts pass and exact same-layer BF16 routers recover at
   least 99.9% of authoritative top-8 assignments.
2. The oracle has zero uncovered LRU52 misses.
3. A lead-2-or-greater arm covers at least 95% of real unique miss groups.
4. Phase74's optimistic latency projection is at or below 4000/65 ms/H4.

Predictor compute is intentionally excluded from gate 4. A passing arm only
authorizes a physical fused-router timing experiment; it is not an E2E speed
claim.
