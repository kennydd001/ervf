# N0R_IDENTITY_REFRESH — report

**Registry:** LIGHTNINGSTREAM_NEMOTRON · **Phase:** `N0R_IDENTITY_REFRESH` (H0)
**Date:** 2026-08-14, started 10:0x UTC · **Preregistration:** `N0R_IDENTITY_REFRESH_PREREGISTRATION_2026-08-14.md`
**Runner:** `scripts/lightningstream_nemotron/n0r_identity_refresh.py`
**Result JSON:** `reports/lightningstream_nemotron/n0r_identity_refresh.json`

## Verdict

**`service_only_unknown_payload`, decision branch 3.** All seven hard gates pass.
This is the outcome the preregistration predicted before execution.

NIM catalog metadata was reachable, but nothing in it binds the served payload to
the five Hugging Face shard LFS digests. No container manifest was inspected — no
credentials were supplied and none were requested. Therefore the two names stay
strictly separated, and everything measured locally from here on is described as
the **public Nemotron 3 Nano NVFP4 checkpoint**, never as "Nemotron 3.5 Lightning
local weights".

## Environment of record

| item | value |
|---|---|
| date/time | 2026-08-14, UTC |
| git commit / dirty | `master`, **no commits exist**; whole tree untracked |
| interpreter | `.venv-nemotron`, Python 3.12.10 |
| `huggingface_hub` | 0.35.3 |
| OS | Windows-11-10.0.26200-SP0 |
| GPU | NVIDIA RTX PRO 2000 Blackwell Laptop, 8,151 MiB, CC 12.0, driver 596.58 — **not used in this phase** |
| shard downloaded | no |
| prompt sent to any endpoint | no |
| protected-80B verification | `PROTECTED_80B_INTACT` (§7) |

## 1. Pinned checkpoint

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`

| field | value |
|---|---|
| requested revision | `ce1b118ae66ec705d02c241525192832eb045fd3` |
| resolved SHA | `ce1b118ae66ec705d02c241525192832eb045fd3` |
| `main` resolves to | `ce1b118ae66ec705d02c241525192832eb045fd3` |
| **drift since N0 (2026-08-12)** | **none** — `main` still equals the pin |
| last modified | 2026-03-15 04:27:10 UTC |
| private / gated / disabled | false / false / false |
| siblings | 18 |
| base model (tag) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `base_model:quantized:` |
| declared languages (tags) | en, es, fr, de, ja, it |

**Dutch is not among the declared languages.** That confirms the assignment's H9
instruction to evaluate Dutch explicitly rather than assume it.

### Shards, with LFS SHA-256

| shard | bytes | LFS SHA-256 |
|---|---:|---|
| `model-00001-of-00005.safetensors` | 3,998,838,864 | `2fdac76b3e4906ce0fb0dd33ab51f011372a5473e0d6c5bb479b6f10d3f29fdb` |
| `model-00002-of-00005.safetensors` | 4,000,414,120 | `559806ee0cb6edcfc01805e24bac9182cb2611bad3993e0da05487d7a79b4f38` |
| `model-00003-of-00005.safetensors` | 3,999,641,680 | `d820849788701123d041501fb8ac88e4ade24a28a63cd663118797cfae910be2` |
| `model-00004-of-00005.safetensors` | 4,000,413,336 | `f5ccb7cfa7870ab2d099134c3f771ad4a158e0421b3bf7b2a0da53311a09cb14` |
| `model-00005-of-00005.safetensors` | 3,343,488,520 | `c9dd9142839367ad274019a7683bc84993217c8a63e70dd8e18656de0c4050eb` |
| **total** | **19,342,796,520** | |

These five digests are the acceptance criterion for the N2 download.

Total shard bytes 19,342,796,520 exceed N1's indexed tensor payload
19,339,781,632 by 3,014,888 bytes — safetensors headers and alignment. Consistent.

Download budget for N2: **18.01 GiB** of payload, against a 25 GiB artifact gate
and 272.57 GiB free disk. Fits.

### Small artifacts pinned byte-exactly

`config.json` (1,817 B), `generation_config.json` (197 B),
`tokenizer_config.json` (188,049 B), `special_tokens_map.json` (420 B),
`model.safetensors.index.json` (2,497,886 B), `hf_quant_config.json` (3,050 B),
plus model code `configuration_nemotron_h.py` (12,893 B),
`modeling_nemotron_h.py` (83,779 B), `nano_v3_reasoning_parser.py` (798 B).
Per-file SHA-256 values are recorded in `n0r_identity_refresh.json`.

## 2. Declared architecture (authoritative for the local checkpoint)

`NemotronHForCausalLM`, `model_type: nemotron_h`, `custom_code`, `transformers_version 4.53.2`.

| field | value |
|---|---:|
| `num_hidden_layers` | 52 |
| `hidden_size` | 2,688 |
| `intermediate_size` / `moe_intermediate_size` | 1,856 |
| `moe_shared_expert_intermediate_size` | 3,712 (= 2 × routed) |
| `n_routed_experts` | 128 |
| **`num_experts_per_tok`** | **6** |
| `n_shared_experts` | 1 |
| `norm_topk_prob` | true |
| `routed_scaling_factor` | 2.5 |
| `n_group` / `topk_group` | 1 / 1 |
| `mlp_hidden_act` | **`relu2`** |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 32 / 2 / 128 |
| `rope_theta`, `partial_rotary_factor` | 10,000, 1.0 |
| `mamba_num_heads` / `mamba_head_dim` | 64 / 64 |
| `ssm_state_size` / `conv_kernel` / `expand` / `n_groups` | 128 / 4 / 2 / 8 |
| `mamba_ssm_cache_dtype` | float32 |
| `chunk_size` | 128 |
| `vocab_size` | 131,072 |
| `tie_word_embeddings` | false |
| **`max_position_embeddings`** | **262,144** |

### Layer pattern

```
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```

52 characters: **23 `M` (Mamba-2), 23 `E` (MoE), 6 `*` (full attention)**.
Attention sits at layer indices 5, 12, 19, 26, 33, 42. The `E` indices are
1, 3, 6, 8, 10, 13, … which **exactly reproduces N1's independently recorded
`moe_layers` list**. Two independent sources agree on the MoE layer set.

## 3. Routing arity — conflict resolved

| source | value |
|---|---|
| pinned public `config.json` | `num_experts_per_tok = 6` |
| N1 header inventory assumption | 6 |
| NIM model card (per assignment) | top-5 in at least one place |

**The pinned config says 6, and N1's derived traffic figures are therefore
sound.** Re-derived here from the config rather than inherited:

- 23 MoE layers × 6 = **138 routed records/token** — matches N1;
- 138 × 5,612,560 = **774,533,280 bytes/token** — matches N1.

The NIM card's top-5 statement is recorded as a **service-side metadata
inconsistency** and has no authority over the local checkpoint. Per the
preregistration, authority for the local model is the pinned config, tensor
index, model code, and one intercepted official routing call — the last of which
is still owed and is scheduled in N3.

## 4. Quantization semantics (declared)

From `hf_quant_config.json`: producer **ModelOpt 0.29.0**,
`quant_algo = NVFP4`, `kv_cache_quant_algo = FP8`, **`group_size = 16`**.

`exclude_modules` holds 63 entries, and the pattern is not arbitrary:

- `lm_head`;
- **every attention layer's** `q_proj`/`k_proj`/`v_proj`/`o_proj` — layers 5, 12, 19, 26, 33, 42, i.e. all six `*` layers;
- the `in_proj`/`out_proj` of the Mamba layer **immediately preceding each attention layer** — layers 4, 11, 18, 25, 32, 41;
- **every** Mamba `conv1d` — all 23 `M` layers.

So all six attention layers are BF16 end to end, six Mamba mixers are BF16, and
every short depthwise convolution stays BF16. These are incompressible fixed
costs that H4's memory plan must carry explicitly; they are not candidates for
the host-resident bank.

## 5. Derived NVFP4 record layout — reproduces every frozen N1 bucket

`scripts/lightningstream_nemotron/n0r_layout_consistency.py` →
`n0r_layout_consistency.json`. Hypothesis: per quantized matrix of `N` weights,
`N/2` bytes of packed 4-bit codes (U8), `N/16` bytes of FP8-E4M3 block scales,
and 2 × FP32 global scales (`weight_scale_2`, `input_scale`).

For a routed expert under `relu2` (two matrices: up `[1856, 2688]`, down `[2688, 1856]`):

| term | bytes |
|---|---:|
| codes, both matrices | 4,988,928 |
| FP8 block scales, both matrices | 623,616 |
| FP32 global scales, 4 × 4 B | 16 |
| **record total** | **5,612,560** |

| quantity | derived | frozen N1 | match |
|---|---:|---:|:--:|
| routed record bytes | 5,612,560 | 5,612,560 | ✅ |
| routed records | 2,944 | 2,944 | ✅ |
| routed bucket bytes | 16,523,376,640 | 16,523,376,640 | ✅ |
| shared bucket bytes | 258,177,392 | 258,177,392 | ✅ |

Residual for trunk after routed+shared: 329,011,200 U8 and 41,126,400 FP8 —
a ratio of exactly 8.000, which is the `(N/2)/(N/16)` signature of group-16
NVFP4, so the trunk's quantized part is internally consistent too.

**Status: `derived_hypothesis_not_layout_proof`.** Four independent byte buckets
reproducing exactly is strong evidence, but it is arithmetic over published
totals. N2 must confirm the layout against real tensor-index entries, dtypes and
offsets before any decoder depends on it.

## 6. NIM endpoint

Five public endpoints were queried without credentials; at least one returned
HTTP 200, so metadata is reachable (branch 3, not branch 4). Nothing returned
binds a served payload to the HF shard digests. `container_manifest` is recorded
as `blocked_no_credentials`; `per_shard_digests_published` is false.

**A second metadata conflict, sharper than the routing one:** the NIM service is
advertised at **1M context**, while the pinned public checkpoint declares
`max_position_embeddings = 262,144`. Under the naming rule these are statements
about two different things and must never be merged. For this research line the
architectural ceiling of the local checkpoint is **262K**, which means the
assignment's `N13_1M_STRETCH` phase cannot be a plain context extension of this
checkpoint — it would require either RoPE scaling beyond the declared maximum or
a different payload. That is recorded now, before any long-context phase is
designed, rather than discovered at N13.

## 7. Gates

| gate | result |
|---|:--:|
| pinned commit resolves | ✅ |
| sibling list complete (18) | ✅ |
| five shards with LFS SHA-256 | ✅ |
| `config.json` parses, architecture extracted | ✅ |
| `num_experts_per_tok` extracted | ✅ |
| routing arity adjudicated against NIM card and N1 | ✅ |
| outcome is a registered value chosen by the frozen rule | ✅ |
| **all gates** | **PASS** |

Protected-80B verification after this phase:
`protected_verification_after_n0r.json` → **`PROTECTED_80B_INTACT`**, root digest
unchanged at `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`,
0 removed / 0 modified / 0 added / 0 listing changes.

## 8. Claim boundary

This phase establishes only what public metadata states today, the exact
architecture declared by the pinned public config, and that payload identity
between the NIM service and the HF checkpoint **cannot** be bound from public
metadata without credentials. It makes no claim about behavioral equivalence,
quality, throughput, or achievable context. The 1M figure is a service-side
statement and is not local evidence. The 29.609 ms transfer floor inherited from
N1 remains a floor, not a latency projection.

## 9. Known limitations

1. No container manifest was inspected; `identity_proven` was never reachable in practice.
2. `behaviorally_close_identity_unproven` was deliberately excluded from this phase because no local reference exists yet, so no prompt suite was run.
3. The layout derivation is arithmetic, not a read of real tensor entries.
4. The `hidden_act` field reported by the runner's generic extractor is `False`; that is a fallback artifact of the extractor's name list, and the authoritative activation is `mlp_hidden_act = "relu2"` read directly from the config. Recorded rather than silently corrected.

## 10. Nearest prior art

`nemotron_h` hybrid Mamba-2/attention MoE is NVIDIA's published architecture with
public modeling code; ModelOpt NVFP4 with FP8 block scales at group 16 is a
published quantization format. Nothing in this phase is novel — it is the
identity and metadata baseline the rest of the line is measured against.

## 11. Next falsification test

`N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS`: download the five official shards into
`models/nemotron_3_5_lightning/`, verify every shard against the LFS SHA-256
recorded above, build an immutable manifest, and **falsify or confirm the §5
layout hypothesis** by reading actual tensor-index entries, dtypes and offsets —
then build a bit-exact decoder for one quantized matrix without materializing a
BF16 model.
