# Phase 20R preregistration

## Identity
Hard target: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, snapshot
e8f3c7c4de75ad84fe1bcef95d38eca76214480b, 52 target layers.

## KV scale audit
Every attention layer must have exactly one scalar k_scale and v_scale; all
must be finite and >0.

On real projected Q/K/V traces, independent FP32 causal attention is the
reference. Compare unit-scale FP8 cache and checkpoint-scale FP8 cache.
Each of six attention layers must have scaled context NRMSE <=0.01 and be
strictly better than unit scale.

## Independent reference
Transformers==5.14.1, isolated venv, use_mamba_kernels=False. The complete
quantized model must execute; AutoConfig recognition is insufficient.

If it executes, run three fixed prompts for 32 greedy target positions each,
storing target ids and top-64 log probabilities.

## Candidate parity
Teacher-force exact reference targets into the patched LightningRuntime.
Required: top1=1, top5=1, mean CE<=0.015, mean coarse KL<=0.010,
p95 KL<=0.040, finite, sabotage observable.

## Gate
PHASE20A_OFFICIAL_PARITY_GREEN only if scale semantics, target consumption,
independent execution, candidate parity and sabotage are all green.
