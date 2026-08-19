# S100 Lightning provenance reset

Date: 2026-08-19

## Material finding

The Phase-14R rerun used the Lightning checkpoint but the quality evaluator still reads `pro_research/results/S100_PHASE3_V18_TRACE_FULL.npz`. That trace has no enforced checkpoint signature in the evaluator. Since the preceding S100 line was run on Nemotron 3 Nano, the reported 74.3% top-1 may primarily measure Lightning-versus-Nano disagreement rather than native-versus-Lightning disagreement.

No Phase-14R native quality conclusion is carried forward until the unchanged Lightning parent is checked against that inherited trace. The inherited trace is quarantined regardless when its metadata lacks a Lightning signature.

## What transfers from Nano

- shapes and memory layouts;
- codec and kernel correctness tests based only on shape/format;
- CUDA launch and arithmetic implementations;
- general experimental harness structure.

## What does not transfer without rerun

- frozen target logits and quality gates;
- routing/cache statistics;
- activation subspace and expert-basis statistics;
- speculative acceptance and DFlash transfer results;
- exact verifier latency and current tok/s claims;
- model-specific threshold selection.

## New numerical hypothesis

The old native path rounded every activation and every projection output to BF16. Phase 15 tests FP32 output and BF16x2 activation decomposition:

```
x_hi = BF16(x)
x_lo = BF16(x - float(x_hi))
[y_hi, y_lo] = TC_GEMM([x_hi, x_lo], W^T, out=FP32)
y = y_hi + y_lo
```

At block B=4 this becomes one M=8 GEMM. The Lightning Phase-14R component already shows that multi-row native BF16 has substantial headroom. The split may recover near-FP32 activation fidelity without rereading the weight matrix.

## Pack

`ervf_s100_lightning_phase15_bf16x2.zip`

SHA256: `22bffffcc8d920930cf581d39f891633a9eb2029d8b39d646e6a9e0b16f36c78`
