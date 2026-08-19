# Phase 20S preregistration

## A — target vs serving metadata

Input: Phase20A schema/audit + local config/hf_quant_config.

Open target-consumption math gate only if:
- the only unknown tensors are exactly the six attention-layer k_scale and
  v_scale scalars;
- quantization metadata explicitly declares FP8 KV-cache;
- no MTP/latent/other target tensor is hidden in unknowns.

## B — FP8 KV serving fidelity

Reference: same LightningRuntime with fp8_kv=False.
Candidate: unchanged LightningRuntime with fp8_kv=True and scale=1.

3 fixed prompts, 64 teacher-forced target positions each.

Serving-use gate:
- top1 >= .99
- top5 = 1
- mean CE <= .01
- mean coarse KL <= .008
- p95 coarse KL <= .03

Failure does NOT close Phase20B; it means Phase20B must keep FP32 KV.

## C — independent layer oracle

Oracle code must not import LightningRuntime loader/dequant kernels.

Capture one real short trajectory with candidate fp8_kv=False.

Compare:
- embedding lookup;
- RMSNorm at sampled layers;
- Mamba mixers: first/middle/last;
- attention mixers: ALL six layers;
- MoE mixers: first/middle/last;
- final RMSNorm + lm_head.

Gates:
- norm NRMSE <= 2e-6
- Mamba mixer output <= 1e-4
- attention mixer output <= 1e-4
- MoE mixer output <= 5e-4
- final logits NRMSE <= 5e-4
- candidate/oracle top1 identical.

## D — optional full Transformers reference

Use isolated Transformers 5.14.1, use_mamba_kernels=False.
Attempt CPU load + Accelerate cpu_offload to CUDA.
Failure is technical if architecture is recognized but quantized kernels or
memory prevent execution.

## Final

PHASE20A_OFFICIAL_PARITY_GREEN=true when:
- target-math consumption gate is green;
- independent layer oracle is green;
- prior target-path sabotage is observable.

A successful full Transformers reference is stronger evidence and is recorded,
but is not mandatory when it is technically impossible and the preregistered
independent layer oracle is fully green.

PHASE20B_FULL_VERIFIER_OPEN=true iff official parity green.

Phase20B must start with fp8_kv=False unless FP8_KV_SERVING_OPEN=true.
