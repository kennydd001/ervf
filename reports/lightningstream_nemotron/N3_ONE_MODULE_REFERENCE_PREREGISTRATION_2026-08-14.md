# N3_ONE_MODULE_REFERENCE — preregistration

**Registry:** LIGHTNINGSTREAM_NEMOTRON · **Phase:** `N3_ONE_MODULE_REFERENCE` (hypothesis H2)
**Date:** 2026-08-14 · **Status at writing:** design frozen, not yet executed.
**Depends on:** `N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS` (PASS, layout confirmed).
**Protected baseline:** root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`.

## 1. Question

Can an independent implementation reproduce the official `nemotron_h` module
semantics — Mamba-2, GQA attention, router, shared expert, routed expert, KV
write/read, and one complete mixed module/residual path — on identical inputs,
within tolerances declared before results are opened?

And, first: **are the two decoder conventions left open by N2 correct?**

## 2. What "official reference" means here, and why

The official `modeling_nemotron_h.py` is plain PyTorch with **no NVFP4 path**:
`NemotronHMLP` builds `nn.Linear` layers expecting real floating-point weights.
Loading the U8/FP8 checkpoint through stock `transformers` therefore cannot work,
and no "run the official model end to end" reference is available at this phase.

The reference is consequently defined as: **the official module classes,
instantiated with weights dequantized by the N2 decoder for quantized modules
and loaded directly for BF16/FP32 modules, executed on CPU in float32.**

That is a real reference for module *math* — it is the publisher's own code path
— but it explicitly does **not** validate the decoder, because both sides then
share it. The decoder is addressed separately in §3.

Two modules escape this limitation entirely and are true official-versus-mine
comparisons, because their weights are stored unquantized:

- the **router** (`gate.weight` F32 `[128, 2688]`, `e_score_correction_bias` F32 `[128]`);
- **attention** (q/k/v/o all BF16, excluded from quantization).

CPU is used deliberately: it removes any possibility of contending with the 80B
agent for the GPU, and N3 makes no performance claim, so no GPU is needed.

## 3. The two open decoder conventions

N2 left `nibble_order = low_first` and the dequant grouping unproven. A finding
that must be stated plainly:

> **Nibble order cannot be falsified by any per-block statistic.** Swapping the
> two nibbles of a byte permutes elements only *within* a byte, and a byte lies
> wholly inside one group of 16. The multiset of values in every block is
> therefore identical under both orders, so block-amax checks, histograms and
> scale-consistency tests are all invariant to it.

It is settled only against an external reference or by end-to-end behavior.
Three attempts, in this order; the first that succeeds decides:

1. **PyTorch native FP4.** If the installed torch exposes `float4_e2m1fn_x2`,
   view the packed U8 buffer as that dtype and convert. That is an authoritative
   unpacking order from an independent implementation.
2. **`nvidia-modelopt`.** If installable offline-safe, use its published
   dequantization helper on one real tensor.
3. **Defer.** If neither is available, record `nibble_order_unresolved`, keep
   `low_first` as the working assumption, and mark the decisive test as
   end-to-end logit coherence at N6. **N3 must not claim the convention is
   confirmed in that case.**

Whichever route resolves it, the check is: decode one real routed-expert matrix
both ways and confirm exactly one matches the external reference bit-for-bit.

## 4. Modules under test and their frozen semantics

Read from `modeling_nemotron_h.py` before writing any implementation, and
recorded here so the implementation cannot quietly drift from it.

### 4.1 Block (`NemotronHBlock.forward`)

Pre-norm, one mixer per layer, single residual:
`residual = h; h = RMSNorm(h); h = mixer(h); return residual + h`.
`residual_in_fp32 = false`, so the residual stays in the input dtype.
`NemotronHRMSNorm` computes variance in float32, applies `rsqrt(var + eps)` with
`eps = layer_norm_epsilon = 1e-5`, multiplies by a float32 weight, then casts
back to the input dtype.

### 4.2 Router (`NemotronHTopkRouter.forward`) — the sharpest risk

```
router_logits = F.linear(h.float32, weight.float32)
scores        = sigmoid(router_logits)
scores_choice = scores + e_score_correction_bias
topk_indices  = topk(scores_choice, k=6, sorted=False).indices
topk_weights  = scores.gather(1, topk_indices)          # NOTE: raw scores
topk_weights /= topk_weights.sum(-1, keepdim=True) + 1e-20
topk_weights *= routed_scaling_factor (2.5)
```

Three details that a naive reimplementation gets wrong:

1. **Selection uses `scores + bias`; weighting uses `scores` without the bias.**
2. `sigmoid`, not `softmax`.
3. With `n_group = topk_group = 1` the group-masking branch is a **no-op**:
   the single group is always selected and `masked_fill` clears nothing. It must
   still be reproduced faithfully in case a future config changes it.

`sorted=False` means the returned index order is **not specified**. Comparisons
must therefore be order-insensitive, and near-ties must be reported.

### 4.3 Routed and shared expert (`NemotronHMLP`)

`down_proj(relu2(up_proj(x)))`, no bias (`mlp_bias = false`). `relu2` is
`ACT2FN["relu2"]`, i.e. `relu(x)**2`. Routed intermediate 1856; shared 3712.

### 4.4 MoE aggregation (`NemotronHMOE`)

```
topk_indices, topk_weights = gate(h)
routed = sum over selected experts of expert(h) * weight   # accumulated in float32
routed = routed.type(h.dtype)
out    = routed + shared_experts(h)                        # shared is NOT gated
```

The shared expert is a **plain add**, not sigmoid-gated. (Qwen3-Next gates its
shared expert; Nemotron does not. The distinction is recorded because the
project's D10 notes describe the gated variant and it must not be carried over.)
Routed accumulation happens in float32 and is cast down **before** the shared
expert is added.

### 4.5 Attention (`NemotronHAttention.forward`) — no positional encoding

q `[4096, 2688]`, k/v `[256, 2688]`, o `[2688, 4096]`; 32 query heads, 2 KV
heads, head_dim 128, `attention_bias = false`. GQA via `repeat_kv` with
`num_key_value_groups = 16`, then `scaled_dot_product_attention` with causal
masking.

> **There is no RoPE.** `apply_rotary_pos_emb` does not appear anywhere in the
> modeling code; `rope_theta` and `partial_rotary_factor` are present in the
> config but unused. The six attention layers are NoPE and positional
> information comes from the 23 Mamba layers.

This is recorded now because it changes later phases: long context here is not a
RoPE-scaling problem, and `N13_1M_STRETCH` cannot be reframed as a RoPE
extension.

### 4.6 Mamba-2 (`NemotronHMamba2Mixer.torch_forward`)

`in_proj [10304, 2688]` splits as `4096 (z) + 6144 (conv: x, B, C) + 64 (dt)`,
where `conv_dim = d_inner + 2 * n_groups * ssm_state = 4096 + 2*8*128`. Depthwise
`conv1d [6144, 1, 4]` with bias, `A_log`/`D`/`dt_bias` per head `[64]`,
`MambaRMSNormGated [4096]` with `group_size`, `out_proj [2688, 4096]`,
`chunk_size = 128`, `mamba_ssm_cache_dtype = float32`.

The `torch_forward` path is used as reference; `cuda_kernels_forward` is out of
scope because no GPU is used.

### 4.7 KV write/read

Exercise a KV cache write and read for one attention layer at two positions and
confirm the read-back equals what was written. The checkpoint declares
`kv_cache_quant_algo: FP8`, but FP8 KV is a **runtime** choice not embodied in
these weights, so N3 records the declaration, tests a BF16 KV round trip, and
additionally measures the round-trip error of an FP8-E4M3 KV store using the N2
codec. No claim is made that the publisher's runtime quantizes KV identically.

## 5. Tolerances — declared before any result is opened

Primary comparisons run in **float32** on CPU, mine versus the official module,
on identical inputs.

| module | metric | gate |
|---|---|---|
| RMSNorm | relative L2 | ≤ 1e-6 |
| router logits | relative L2 | ≤ 1e-6 |
| router top-6 **index set** | exact set equality | required |
| router top-6 weights (matched by index) | max abs | ≤ 1e-6 |
| routed expert | relative L2 | ≤ 1e-5 |
| shared expert | relative L2 | ≤ 1e-5 |
| MoE aggregate | relative L2 | ≤ 1e-5 |
| attention | relative L2 | ≤ 1e-5 |
| Mamba-2 | relative L2 | ≤ 1e-4 |
| full mixed block + residual | relative L2 | ≤ 1e-4 |
| KV BF16 round trip | exact equality | required |

`relative L2` is `‖mine − ref‖₂ / ‖ref‖₂`. Mamba-2 and the composite path get a
looser bound because they contain `exp`, cumulative sums and a chunked scan whose
association order is not required to match.

Per the assignment, **cross-backend and whole-sequence bit identity are not
demanded**. BF16-path results are reported for information and are **not** gated.

A near-tie in router selection is not automatically a failure: if an index set
mismatch occurs, the margin between the 6th and 7th bias-corrected score is
reported, and a mismatch with margin below 1e-6 is classified
`tie_ambiguous` rather than `wrong`. Any mismatch with a larger margin is a
failure. This rule is fixed now so it cannot be invented after seeing a result.

## 6. Inputs

Deterministic and recorded: a fixed seed, one frozen pseudo-random hidden-state
batch in float32 of shape `[1, 8, 2688]` scaled to a realistic RMS, plus the
real weights of layer 1 (MoE), layer 0 (Mamba) and layer 5 (attention). Using
synthetic activations is deliberate — N3 tests module *semantics*, and no
natural activation capture exists until a full forward runs at N6. The route IDs
captured here are therefore **synthetic-input routes** and must never be
described as natural routes.

## 7. Hard gates

1. every module in §4 has an independent implementation and a comparison result;
2. every tolerance in §5 met, or the phase fails;
3. the router top-6 index set matches exactly (or is `tie_ambiguous` per §5);
4. one official router call is captured and persisted for reuse;
5. the nibble-order question reaches one of `confirmed`, `falsified`, or
   `unresolved_deferred_to_N6` — and if unresolved, N3 claims nothing about it;
6. no GPU is used; no timing figure is produced;
7. no protected byte changed.

## 8. Stop rules

- A module cannot be reproduced within tolerance → **debug the implementation
  only**. Make no statement about the checkpoint or the model's quality.
- The router index set mismatches with a non-trivial margin → stop and diagnose;
  do not adjust the tolerance.
- The nibble order is falsified → N2's decoder assumption is wrong; annotate
  N2 rather than rewriting it, fix the codec, and re-run N2's decoder validation.

## 9. Claim boundary

N3 may claim only that an independent implementation reproduces the official
module math on synthetic inputs within declared tolerances, on CPU, in float32,
for one layer of each type. It may **not** claim full-model correctness, natural
routing behavior, quality, latency, throughput, memory feasibility, or that the
decoded weights equal the publisher's BF16 source weights.

## 10. Artifacts

| path | kind |
|---|---|
| `src/moe_lab/lightningstream_nemotron/reference.py` | independent module implementations |
| `src/moe_lab/lightningstream_nemotron/loader.py` | range-read weight loader and dequant |
| `scripts/lightningstream_nemotron/n3_module_reference.py` | runner |
| `reports/lightningstream_nemotron/n3_module_reference.json` | comparison results |
| `reports/lightningstream_nemotron/n3_nibble_order_resolution.json` | §3 outcome |
| `reports/lightningstream_nemotron/n3_official_route_capture.json` | captured official top-6 call |
| `reports/lightningstream_nemotron/N3_ONE_MODULE_REFERENCE_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n3_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n3.json` | protected check |

## 11. Non-interference

CPU-only and single-process. A live-process check runs before execution; because
no GPU is touched, N3 may proceed even if the 80B agent is computing.
