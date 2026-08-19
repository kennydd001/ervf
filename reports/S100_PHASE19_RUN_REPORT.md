# S100 Phase 19 — Nemotron 3.5 Lightning residual projection

Datum: 2026-08-19  
Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`  
Snapshot: `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`  
Config: 52 layers, hidden size 2688, `NemotronHForCausalLM`  
Scripts: `pro_research/s100_phase19_residual_projection.py`, `pro_research/s100_phase19_full_layer.py`

## Model identity correction

This run used the actual 3.5 Lightning snapshot. It did not use the earlier
Nano checkpoint. On the real 3.5 checkpoint, Mamba in/out projections in the
sampled layers 0, 25, and 50 are all `fp8_tensor`. The BF16/NVFP4 split seen in
the earlier Nano run therefore does not describe this model.

## Residual projection screen

One BF16 activation term leaves approximately `1.6e-3` to `1.7e-3` NRMSE and
fails the `1e-4` gate. Two terms,

```text
x0 = BF16(x)
r1 = x - x0
x1 = BF16(r1)
y  = W*x0 + W*x1
```

pass on every sampled layer, side, and H=4 case at approximately `2.4e-6` to
`2.5e-6` NRMSE. The batched FP8 block candidate reaches these H=4 speedups:

| layer | in projection | out projection |
|---:|---:|---:|
| 0 | 1.501x | 1.788x |
| 25 | 1.372x | 1.518x |
| 50 | 1.418x | 1.615x |

Three terms reduce the residual error further to roughly `1.3e-7`, but are not
needed for the stated correctness gate.

## Full H=4 Mamba-layer retest

The two-term residual path was fused into one FP8 block kernel. The Phase17
block convolution, affine SSM scan, and gated norm core were retained. The
comparison uses the exact captured recurrent/KV starting state and exact
production projection as the baseline.

| layer | full-layer speedup | output NRMSE | conv-state NRMSE | SSM-state NRMSE | gate |
|---:|---:|---:|---:|---:|---|
| 0 | 1.729x | 1.253e-6 | 1.532e-6 | 3.788e-7 | pass |
| 25 | 2.067x | 3.764e-6 | 2.162e-6 | 3.070e-7 | pass |
| 50 | 2.014x | 2.004e-6 | 1.792e-6 | 4.173e-7 | pass |

All three layers pass the output, convolution-state, and SSM-state gates.

Therefore:

```text
PHASE20_FULL_BLOCK_VERIFIER_OPEN = True
```

This opens the next research step, but it is not yet an end-to-end decode or
100-token/s claim. The current result is a real H=4 Mamba-layer ceiling on the
correct 3.5 Lightning checkpoint.

## Important implementation boundary

The 3.5 result uses FP8 Mamba projections, not NVFP4 Mamba projections. The
NVFP4 scaled-MM route proposed for the Nano layout was not silently substituted
here. The tested FP8 residual kernel preserves the original FP8 weight bytes
and scalar weight scale, and computes the two BF16 activation residual terms
inside the batched projection kernel.

Machine-readable outputs:

- `pro_research/results/s100_phase19_residual_projection.json`
- `pro_research/results/s100_phase19_full_layer.json`
