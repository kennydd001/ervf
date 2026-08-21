# S100 Phase84 — integrated target-only H4 correctness gate

## Frozen question

Can one continuous custom target path execute an authoritative four-token
target block through all 40 Ornith layers and the ERVF head while preserving
independent HF-FP8 state, route and logit parity?

## Method

- Use tokens 0..3 from the committed target-only 64-token reference trace.
- Start every linear-attention convolution/recurrent state and every
  full-attention KV cache from zero.
- Execute real embedding, 40 dense attention blocks, routed top-8 and shared
  NVFP4 experts, residuals, final norm, native head shortlist and exact ERVF
  rerank.
- Quantize FP8 inputs as `E4M3(x / input_scale)`.
- Derive routes from the custom target hidden state; no captured route is fed
  into execution.
- Maintain a separate CPU HF-ModelOpt dequant reference from embedding through
  all layers. Compare every layer's residual and routes, all recurrent/KV
  states, and the complete final logits.

## Gates

1. All 40 route sets match the independent target reference exactly.
2. Every layer residual is finite and the final residual NRMSE is at most 2e-3.
3. Linear recurrent/conv state and full-attention KV state parity are finite and
   within the same 2e-3 NRMSE envelope.
4. Complete final logits are finite with NRMSE at most 2e-3 and matching top-1.
5. Native shortlist plus exact ERVF rerank returns the exact control top-1 for
   all four rows.

This gate does not measure output tok/s. Timing starts only after the independent
CPU reference, disk loading and parity synchronizations are removed from the
continuous GPU epoch.
