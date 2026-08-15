# N3_ONE_MODULE_REFERENCE — report

**Registry:** LIGHTNINGSTREAM_NEMOTRON · **Phase:** `N3_ONE_MODULE_REFERENCE` (H2)
**Date:** 2026-08-14 · **Preregistration:** `N3_ONE_MODULE_REFERENCE_PREREGISTRATION_2026-08-14.md`
**Depends on:** `N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS` (PASS)

## Verdict

**PASS. All twelve gates satisfied, every tolerance met with two to four orders
of margin — and the nibble-order question left open by N2 is CONFIRMED against an
independent published implementation.**

## Environment of record

| item | value |
|---|---|
| date | 2026-08-14 |
| git commit / dirty | `master`, no commits exist; tree untracked |
| interpreter | `.venv-nemotron`, Python 3.12.10 |
| libraries | torch 2.9.1+cpu, transformers 4.53.2, numpy 2.2.6, torchao 0.18.0 |
| device | **CPU only** — `torch.cuda.is_available() == False` (CPU build) |
| model / revision | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` @ `ce1b118a…` |
| seed / seq_len | 20260814 / 8 |
| layers exercised | MoE 1, Mamba 0, attention 5 |
| timing produced | none |
| protected-80B check | `PROTECTED_80B_INTACT` |

CPU was chosen deliberately: N3 makes no performance claim, and a CPU-only torch
build makes GPU contention with the 80B agent structurally impossible rather than
merely unlikely.

## 1. Nibble order — CONFIRMED as `low_first`

N2 flagged this as its most dangerous open assumption. A finding worth stating
in its own right:

> **Nibble order cannot be falsified by any per-block statistic.** Swapping a
> byte's two nibbles permutes elements only *within* that byte, and a byte lies
> wholly inside one group of 16. Every block's value multiset is therefore
> identical under both orders, so block-amax checks, histograms and
> scale-consistency tests are all blind to it.

It was resolved against an external reference instead, in the preregistered
order:

| attempt | source | result |
|---:|---|---|
| 1 | `torch.float4_e2m1fn_x2` | dtype exists and the buffer views cleanly, but conversion raises `NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'` on CPU |
| 2 | **torchao 0.18.0** `unpack_uint4` + `f4_unpacked_to_f32` | **decides** |

On **2,097,152 real elements** of `backbone.layers.1.mixer.experts.0.up_proj.weight`:

| order | codes match | decoded values match |
|---|:--:|:--:|
| `low_first` | ✅ | ✅ |
| `high_first` | ❌ | ❌ |

The two orders were verified to produce genuinely different output on this data,
so the match is discriminating rather than degenerate. torchao's `pack_uint4` is
`data[::2] | data[1::2] << 4` — element `2i` in the low nibble — which is exactly
the convention the N2 codec implemented.

Reading torchao's `NVFP4Tensor.dequantize` also confirms the second open
assumption structurally: it unpacks, converts E2M1, reshapes to
`[M, K/block, block]` and multiplies by the per-block scale **along the
contraction dimension**, with the per-tensor scale folded in — matching the N2
decoder and the N2 finding that grouping runs along the contraction axis.

## 2. Module comparisons — all within tolerance

Independent numpy implementations versus the checkpoint's own
`modeling_nemotron_h.py`, identical inputs, CPU, float32 in / float64
accumulation. Tolerances were fixed in the preregistration before any result was
opened.

| module | relative L2 | tolerance | margin | result |
|---|---:|---:|---:|:--:|
| RMSNorm | 6.837e-08 | 1e-06 | 15× | ✅ |
| router logits | 2.519e-07 | 1e-06 | 4× | ✅ |
| routed expert (`up → ReLU² → down`) | 3.008e-07 | 1e-05 | 33× | ✅ |
| shared expert | 1.746e-07 | 1e-05 | 57× | ✅ |
| MoE aggregate | 1.886e-07 | 1e-05 | 53× | ✅ |
| attention (GQA, NoPE) | 7.666e-07 | 1e-05 | 13× | ✅ |
| **Mamba-2** | **5.435e-07** | 1e-04 | **184×** | ✅ |
| mixed block + residual | 1.095e-07 | 1e-04 | 913× | ✅ |

### The Mamba-2 result is the strongest of these

My implementation is a **plain sequential recurrence**; the official
`torch_forward` is a **chunked SSD factorisation** with `segment_sum`, cumulative
decay and inter-chunk state propagation. These are different algorithms, not two
copies of one. Agreement to 5.4e-07 across the full 8-token sequence is therefore
real evidence that the SSD semantics are understood — including the `in_proj`
split `4096 (z) + 6144 (conv) + 64 (dt)`, the depthwise causal conv with
`kernel=4`, `softplus(dt + dt_bias)` with the `(0, inf)` clamp,
`A = -exp(A_log)`, per-head decay, group→head broadcast at 8 heads per group, and
the `D` skip on the pre-`dt` hidden states.

### Router — the sharpest risk, cleared

| check | result |
|---|---|
| top-6 index set, all 8 tokens | **exact match** |
| weights matched by expert id, max abs diff | 1.099e-07 (tol 1e-6) |
| minimum tie margin across tokens | **1.192e-03** |

The tie margin matters: the preregistration allowed a `tie_ambiguous`
classification for mismatches below 1e-6. The smallest observed margin is over a
thousand times larger than that threshold, so the index agreement is genuine and
did not depend on the escape clause.

The three traps were reproduced deliberately: selection on `scores + bias` but
weighting on **raw scores**; `sigmoid` rather than `softmax`; and the group-mask
branch, which is a no-op at `n_group = topk_group = 1` but is implemented anyway
so a future config change cannot silently break it.

One official router call was captured and persisted to
`n3_official_route_capture.json`. It is labelled **synthetic-input routes** in the
artifact itself — these are not natural routes and must never be described as
such.

### KV cache

BF16 write/read round trip is **exact**. The checkpoint declares
`kv_cache_quant_algo: FP8`; an FP8-E4M3 store of the real K tensor was measured
at relative L2 **2.454e-03**. That figure is indicative only — FP8 KV is a
runtime choice not embodied in these weights, and nothing here claims the
publisher's runtime quantizes KV this way.

## 3. Shims — what was patched and what it costs

Two in-process shims. No file on disk was modified.

**(a) `torch.nn.functional` symbols + CUDA stream context.** torchao — installed
solely as the external FP4 reference — is imported by
`transformers.modeling_utils` and expects torch ≥ 2.11, so `ScalingType`,
`SwizzleType` and friends were defined locally. Separately,
`NemotronHBlock.forward` wraps its body in
`torch.cuda.stream(torch.cuda.default_stream(device))`, which hard-fails on a
CPU-only build; it was replaced with a null context. The official comment
describes it as a multi-GPU NaN guard — it is scheduling, not arithmetic. **No
correctness cost, and no torchao kernel is invoked by this runner.**

**(b) `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`.** The official module
raises `ImportError` without `mamba_ssm`, which requires CUDA and cannot be
installed here. Our gated RMSNorm was supplied instead.

> **This one has a real cost and is recorded as such:** the gated RMSNorm is used
> by **both** sides of the comparison, so it is **not independently validated by
> N3**. Everything else in the Mamba mixer — projection, conv, split, dt, the
> entire SSD scan, the output projection — is validated, because the scan is the
> part where the two implementations genuinely differ. The gated norm's
> correctness rests on mamba_ssm's documented `norm_before_gate=False` semantics
> (gate first, then grouped RMS over blocks of 512, then the full-width weight)
> and is deferred to end-to-end coherence at N6.

## 4. Architectural findings recorded for later phases

### 4.1 There is no RoPE

`apply_rotary_pos_emb` appears **nowhere** in `modeling_nemotron_h.py`.
`rope_theta` and `partial_rotary_factor` are present in the config but unused.
The six attention layers are **NoPE**; all positional information comes from the
23 Mamba-2 layers.

Consequence, recorded now rather than at N13: long context here is **not** a
RoPE-scaling problem. `N13_1M_STRETCH` cannot be reframed as a RoPE extension of
this checkpoint, whose declared ceiling is 262,144.

### 4.2 The shared expert is ungated

`NemotronHMOE.forward` computes `routed + shared_experts(h)` — a plain add.
Qwen3-Next gates its shared expert with a sigmoid, and the project's D10 notes
describe that gated variant. **That must not be carried over.** Routed
contributions accumulate in float32 and are cast to the input dtype **before**
the shared expert is added.

### 4.3 Per-sequence state is remarkably small

Derived from real tensor shapes in `n3_state_budget.json` — **arithmetic
projections, not measured allocations**:

| context | KV (FP8) | KV (BF16) | + Mamba state |
|---:|---:|---:|---:|
| 4,096 | 12.0 MiB | 24.0 MiB | 59.1 MiB |
| 32,768 | 96.0 MiB | 192.0 MiB | 143.1 MiB |
| 131,072 | 384.0 MiB | 768.0 MiB | 431.1 MiB |
| 262,144 | 768.0 MiB | 1536.0 MiB | 815.1 MiB |

KV is only 3,072 elements per token because just six layers carry one, each with
two KV heads. Mamba state is **47.078 MiB and constant in context length**
(48,234,496 B of FP32 SSM state + 1,130,496 B of BF16 conv state).

For scale: the project's STREAMQ5 Qwen3-30B runtime needed 402,653,184 B of KV
for **4K** context. This checkpoint's entire per-sequence state at **262K** is
about 2× that — a 64× context ratio for roughly double the state. That is the
strongest architectural argument for this target so far, and it is now computed
from real shapes rather than from a model card.

It is a projection about state bytes only. It says nothing about attention
compute over 262K positions, about whether the expert bank streams fast enough,
or about achievable tokens per second.

## 5. Gates

| gate | result |
|---|:--:|
| RMSNorm within 1e-6 | ✅ |
| router logits within 1e-6 | ✅ |
| router top-6 index set exact | ✅ |
| router weights within 1e-6 | ✅ |
| routed expert within 1e-5 | ✅ |
| shared expert within 1e-5 | ✅ |
| MoE aggregate within 1e-5 | ✅ |
| attention within 1e-5 | ✅ |
| Mamba-2 within 1e-4 | ✅ |
| mixed block within 1e-4 | ✅ |
| KV BF16 round trip exact | ✅ |
| no GPU available, so none used | ✅ |
| nibble order reaches a registered resolution | ✅ (**confirmed**) |
| one official router call captured | ✅ |
| no protected byte changed | ✅ |

*Process note:* the runner initially carried a `no_gpu_used` gate written as
`not cuda_available or True`, which is vacuously true. It was replaced with
`not torch.cuda.is_available()`, which is meaningful on this CPU-only build, and
the phase was re-run. A gate that cannot fail is not a gate.

## 6. Claim boundary

Established: independent implementations reproduce the official `nemotron_h`
module math on **synthetic** inputs, within tolerances declared in advance, on
CPU in float32, for one layer of each type; and the NVFP4 packing order matches an
independent published implementation.

**Not** established: full-model correctness, natural routing behavior, any
quality result, any latency or throughput figure, memory feasibility, or that the
decoded weights equal the publisher's BF16 source weights. The captured routes
are synthetic-input routes. The state table in §4.3 is a projection.

## 7. Known limitations

1. Gated RMSNorm is shared by both sides and therefore unvalidated (§3b).
2. Synthetic activations, not natural ones — no natural activation exists until a full forward runs.
3. One layer of each type, one 8-token sequence, batch 1.
4. The Mamba comparison used the prefill path with no cache; the cached decode branch (`cache_position > 0`) is a different code path and was **not** exercised.
5. FP8 KV is indicative only.
6. `float4_e2m1fn_x2` has no CPU conversion kernel, so attempt 1 could not decide the nibble question; a GPU build might make it a second independent witness.

## 8. Nearest prior art

`nemotron_h` is NVIDIA's published architecture with public modeling code; the
Mamba-2 SSD recurrence is from the published Mamba-2 work; NVFP4 packing is a
published format with reference implementations in torchao. Nothing here is
novel — the value is that the semantics every later phase depends on are now
verified against the publisher's own code rather than assumed.

## 9. Next falsification test

`N4_ZERO_CACHE_DATAPLANE`. Build the routed-expert path with **no expert cache**
— the cleanest active-set baseline — using the three-range random-access shape
measured in N2 (4,988,928 B codes + 623,616 B scales + 16 B globals per expert,
with a two-file gather for the four boundary-straddling experts). Screen against
the frozen gates: routed-expert path p95 ≤ 45 ms and zero output mismatches
versus the NVFP4 target semantics, with no full dequantized expert materialised.

Inherited context: N1's all-cold floor is 29.609 ms/token at 26.158915 GB/s, and
PORT80B's diagnosis says per-record dispatch and a host gather pass — not PCIe
bandwidth — are what actually cost the time. With 138 records/token instead of
480, registered batched copies should be tested before any cache design.
