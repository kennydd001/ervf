# Phase 20S handoff

Do not patch k_scale/v_scale into production in this phase.

Primary distinction:

TARGET MODEL MATH
  NemotronH attention computes Q/K/V -> causal attention -> O.
  k_scale/v_scale are absent from the Transformers NemotronH module math.

QUANTIZED SERVING CACHE
  vLLM/ModelOpt may load k_scale/v_scale when FP8 KV cache is explicitly
  enabled. This is an implementation/quantization policy.

Therefore:
- Phase20B correctness target uses `fp8_kv=False`.
- FP8 KV is adjudicated separately.
- The 12 scale tensors may be intentional serving metadata only when the
  local quantization config declares an FP8 KV-cache scheme.
- Any other unconsumed tensor still fails closed.

The NumPy oracle intentionally contains its own safetensors reader and its own
BF16, E4M3 and NVFP4 decoding. It may import LightningRuntime only to capture
real layer inputs/candidate outputs; it must not import loader/fused_nvfp4
math to compute the oracle output.
