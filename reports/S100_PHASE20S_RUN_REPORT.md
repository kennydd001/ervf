# S100 Phase 20S — Target Math + Independent Oracle

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`  
Snapshot: `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`

## Final adjudication

- `TARGET_MATH_CONSUMPTION_GREEN`: **True**
- `INDEPENDENT_LAYER_ORACLE_GREEN`: **True**
- `FP8_KV_SERVING_OPEN`: **False** (technical pinned-memory failure; not a target-math blocker)
- Phase 20B KV policy: **`fp8_kv=False`**
- `PHASE20A_OFFICIAL_PARITY_GREEN`: **True**
- `PHASE20B_FULL_VERIFIER_OPEN`: **True**
- S100 achieved: **False** — no full H=4 block timing was run in Phase 20S.

The initial reclassification had a pack naming bug. The checkpoint keys are
`mixer.k_proj.k_scale` and `mixer.v_proj.v_scale`; after correcting that exact
name match, all twelve tensors were classified as optional FP8-KV serving
metadata and no target-math tensor remained unknown.

The independent NumPy/safetensors layer oracle passed norm, Mamba, attention,
MoE, final norm, final logits and final top-1 gates. The official Transformers
5.14.1 full-model load was attempted but technically blocked by quantized
shape/ModelOpt mismatches. The independent layer oracle is therefore the
accepted fallback evidence.

Next route: build the full 52-layer H=4 perfect-draft verifier with
`fp8_kv=False`.
