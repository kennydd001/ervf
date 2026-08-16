# S100 native NVFP4 C3B — real activation W4A4 preregistration

Date frozen: 2026-08-16
Branch: `pro-s100-nativefp4-c2b`
Parent evidence: C3A-v2 real checkpoint representation/geometry PASS; activation-side recipe sweep closes W4A8.

## Question

When **real V18 decode activations** are dynamically quantized to NVIDIA NVFP4 (E2M1 data, E4M3 block scale per 16 values, F32 global scale), does the native SM120 FP4 path preserve projection/logit quality closely enough to justify integration work, and what does **activation quantization + GEMM together** cost at M in {1,2,4,8}?

C3B is deliberately not an end-to-end adoption claim. It isolates the activation approximation that C3A excluded.

## Frozen prerequisites

1. `C3A_V2_LAYOUT_PREFLIGHT.json` is `layout_v2_preflight_pass`.
2. `C3A_REAL_WEIGHT.json` is correctness-green (`real_weight_representation_and_geometry_candidate` or `real_weight_representation_green_perf_miss`).
3. The activation-side recipe sweep reports `w4a4_only`: BF16/FP8 A is not a valid mixed-precision replacement for the BlockWise1x16 NVFP4 B path.

If any prerequisite is absent or contradictory, C3B fails closed.

## Capture protocol

The capture runs in `.venv-nemotron` with the V18 arithmetic stack installed:

- capacity policy from V18;
- device-resident routing/cache;
- selective dense ERVF;
- batched routed-up;
- H-SCALE + B3 combined MoE path when its preregistered plane allocation fits.

CUDA graph replay is intentionally **not** used during capture: eager submission exposes intermediate tensors without changing V18 arithmetic. V18 itself is already registered bit-exact against its baseline; C3B does not time the capture process.

Prompt set is frozen to `graph_e1f22._load_prompt_set("full")`: the two anchor prompts plus the code prompt. Exactly 8 decode positions per prompt are captured (24 rows total). The target MoE layer is the first routed layer and must be layer 1, matching C3A's representative checkpoint tensors.

For each row C3B stores float32 binary arrays and SHA256 metadata for:

- `moe_normed`: input shared by `shared_up` and all routed `up_proj` experts (K=2688);
- `shared_up_ref`: current W4A32 raw linear output (N=3712);
- `shared_down_input`: the real ReLU² shared activation (K=3712);
- `shared_down_ref`: current W4A32 raw output (N=2688);
- `routed_up_ref`: expert-0 current W4A32 raw output (N=1856), evaluated out-of-band on the captured `moe_normed` without mutating model state;
- `lm_head_input`: final normalized hidden state (K=2688);
- `lm_head_ref`: current W4A32 full logits (N=131072).

The extra W4A32 reference GEMVs happen only after the original target MoE computation and write dedicated scratch buffers; they are not performance measurements and are not fed back into the model.

## Frozen activation quantizer

C3B mirrors TorchAO's traditional two-level NVFP4 algorithm:

- `per_tensor_scale = amax / (448 * 6)`;
- per-16 block `block_scale = block_amax / 6`;
- quantize `block_scale / per_tensor_scale` to `float8_e4m3fn`, clamped to `[tiny(E4M3), 448]`;
- normalize high-precision values by the combined global + local scale;
- clamp to `[-6, 6]`;
- convert to E2M1 with round-to-nearest, ties-to-even;
- pack even-K nibble low, odd-K nibble high;
- swizzle the block scales with the corrected TorchAO row-block-major `to_blocked` layout proven by C3A-v2.

No TorchAO package is required at runtime; the dependency-light implementation vendors only these frozen arithmetic/layout rules and cites the upstream source in comments.

The global activation scale is recomputed per M-batch, matching dynamic per-tensor NVFP4 semantics.

## M grouping

Quality is measured for M={1,2,4,8}. Batches never cross prompt boundaries. Since every prompt has exactly 8 captured decode positions, each M partitions every prompt exactly.

This is a numerical batching experiment only. It does **not** claim that eight causally dependent positions can already be scheduled together in autoregressive decode.

## Correctness / quality gates

All thresholds are frozen before observing C3B results.

- `C3B_G1_capture_integrity`: 3 prompts x 8 rows, expected dimensions, finite capture references, binary SHA256 match.
- `C3B_G2_parents_green`: C3A-v2 representation parents green and W4A8 closed.
- `C3B_G3_activation_reuse_identity`: `shared_up` and `routed_up` use the identical captured `moe_normed` rows; no second quantizer is allowed in the cost model for routed-up.
- `C3B_G4_native_executes`: every family and every M executes finite native BF16 output.
- `C3B_G5_activation_quant_quality`: for every family/M aggregate activation cosine >= 0.995 and normalized RMSE <= 0.120.
- `C3B_G6_projection_quality`: for every family/M aggregate output cosine >= 0.995 and normalized RMSE <= 0.100.
- `C3B_G7_projection_max_error`: for every family/M normalized max-absolute error <= 0.250.
- `C3B_G8_lm_top1`: lm-head top-1 agreement with the W4A32 reference >= 0.95 at every M (>=23/24 rows).
- `C3B_G9_lm_top5_overlap`: mean top-5 overlap >= 0.80 at every M.
- `C3B_G10_lm_distribution`: mean KL(reference || native) <= 0.020 nat and max KL <= 0.100 nat at every M.

A failure of G5-G10 closes the **direct PTQ W4A4 activation route** at these thresholds. It is not relabeled as a kernel bug. QAT/static-scale variants would be separate hypotheses.

## Performance protocol

Performance is secondary to quality and is measured in `.venv-fp4-c2b` on the same captured inputs.

For each family and M={1,2,4,8}, record:

1. prototype dynamic activation quantizer-only p50;
2. prequantized native GEMM p50;
3. **combined dynamic quantizer + native GEMM p50**.

The combined arm rotates enough real B copies to make the B working set >=4.0x L2, with no fixed copy cap. If VRAM cannot satisfy that, the family is `not_run_memory_gate` and cannot count as a performance pass.

The current quantizer is a dependency-light Torch implementation of the upstream algorithm, not a claimed optimal fused decode kernel. Therefore performance gates are engineering signals, not quality closure gates:

- `C3B_P1_cold_honest`: every measured combined arm has B rotation >=4x L2;
- `C3B_P2_M8_total_over_M1_le_2`: combined M8 total latency <=2.0x M1 for at least 3/4 families;
- `C3B_P3_reuse_cost_model`: weighted quantizer accounting charges `moe_normed` once per MoE layer and reuses it for shared-up + six routed-up calls.

A performance miss means a fused/specialized activation quantizer is required; it does not invalidate a quality pass.

## Claim boundary

A green C3B proves only that, on 24 frozen real V18 decode states:

1. dynamic NVFP4 activation PTQ meets the preregistered local projection/logit gates;
2. the real checkpoint native FP4 kernel continues to execute correctly with nontrivial real activations;
3. quantizer + GEMM cost has been measured together under honest B-cache rotation;
4. activation quantization reuse across shared/routed up projections is available in the cost model.

It does **not** prove full-rollout parity, CE/perplexity over a corpus, causal M=8 scheduling, grouped-MoE native integration, or end-to-end token/s. Those belong to C3C / a full rollout quality gate.
