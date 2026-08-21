# S100 Phase63: Ornith NVFP4 LM-head H4 preregistration

## Question

Does the Nemotron direct-L2 H4 LM-head technique transfer to Ornith's real
248,320 x 2,048 NVFP4 head, avoiding four reads of its 272.8 MiB payload?

## Arms

- Control: four production ERVF H1 GEMVs.
- `warp32_m4`: the exact-reduction Phase49 M4 kernel.
- `r16_m4`: the Phase31 selected direct-L2 M4 occupancy kernel, with sixteen
  output rows per CTA.

All arms use identical random finite F32 inputs and real Pottokao LM-head bytes.
The control is run once before both candidates. CUDA-event medians after warmup
are decisive.

## Gates

1. Both candidates are finite and deterministic.
2. Both preserve all four control argmax token IDs.
3. Both have normalized RMSE <= 0.001 and cosine >= 0.999999 versus control.
4. The selected fastest correct M4 arm is at least 2.5x faster than H1 x4.
5. The selected arm has zero local memory and at most 64 registers/thread.

This measures full H4 logits but excludes final norm and argmax latency.
