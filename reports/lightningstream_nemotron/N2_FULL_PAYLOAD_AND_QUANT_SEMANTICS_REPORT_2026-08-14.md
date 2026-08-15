# N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS — report

**Registry:** LIGHTNINGSTREAM_NEMOTRON · **Phase:** `N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS` (H1)
**Date:** 2026-08-14 · **Preregistration:** `N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS_PREREGISTRATION_2026-08-14.md`
**Depends on:** `N0R_IDENTITY_REFRESH` (`service_only_unknown_payload`)

## Verdict

**PASS. All frozen gates satisfied. The N0R layout hypothesis is CONFIRMED
against real tensor entries.**

The local copy is byte-identical to the published checkpoint, the on-disk
quantization layout is exactly the ModelOpt NVFP4 convention derived in N0R, and
one routed expert's two matrices decode self-consistently under published NVFP4
semantics with a bit-exact round trip over 9,977,856 codes.

## Environment of record

| item | value |
|---|---|
| date | 2026-08-14 |
| git commit / dirty | `master`, no commits exist; tree untracked |
| interpreter | `.venv-nemotron`, Python 3.12.10, numpy 2.2.6, huggingface_hub 0.35.3 |
| OS / CPU | Windows 11 26200 · Intel Core Ultra 9 285H, 16 cores |
| GPU | **not used in this phase** |
| model id / revision | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` @ `ce1b118ae66ec705d02c241525192832eb045fd3` |
| payload decoded | one routed expert only |
| BF16 model materialized | no |
| protected-80B check | `PROTECTED_80B_INTACT` |

## 1. Download and integrity

All five shards verified **on the first attempt**; no redownload was needed.

| shard | bytes | SHA-256 verified |
|---|---:|:--:|
| `model-00001-of-00005.safetensors` | 3,998,838,864 | ✅ |
| `model-00002-of-00005.safetensors` | 4,000,414,120 | ✅ |
| `model-00003-of-00005.safetensors` | 3,999,641,680 | ✅ |
| `model-00004-of-00005.safetensors` | 4,000,413,336 | ✅ |
| `model-00005-of-00005.safetensors` | 3,343,488,520 | ✅ |
| **total** | **19,342,796,520** (18.014383 GiB) | |

Artifact gate ≤25 GiB: **pass** (18.014 GiB). Ten small companions (config,
tokenizer, index, quant config, model code) were placed beside the shards so the
directory is self-contained.

### Independent provenance cross-check

Shard 1's header reproduces the values N1 recorded two days earlier from the
*remote* repository: 429,488 header bytes, 3,454 tensors, payload extent
3,998,409,368, and SHA-256
`f9b2428248cfb2b8d36dbd879882f72e1a9ed417d4a734c65d80c7192c5a1a78`.

That last one initially mismatched. The cause was a hashing-convention
difference, not a data difference: **N1 hashed the 8-byte little-endian length
prefix together with the header body**, while hashing the body alone gives
`da994537…`. Both conventions are now recorded per shard so the ambiguity cannot
recur, and the N1 comparison is made like-for-like.

## 2. Tensor inventory — every N1 bucket reproduced exactly

| quantity | measured locally | N1 frozen | match |
|---|---:|---:|:--:|
| tensors | 24,147 | 24,147 | ✅ |
| tensor bytes | 19,339,781,632 | 19,339,781,632 | ✅ |
| routed experts | 16,523,376,640 | 16,523,376,640 | ✅ |
| shared experts | 258,177,392 | 258,177,392 | ✅ |
| trunk / other | 2,558,227,600 | 2,558,227,600 | ✅ |
| routed records | 2,944 | 2,944 | ✅ |
| routed record bytes | 5,612,560 (uniform, all 2,944) | 5,612,560 | ✅ |

dtype split, also exactly matching N1:

| dtype | bytes | tensors |
|---|---:|---:|
| U8 (NVFP4 codes) | 15,245,905,920 | 5,968 |
| BF16 | 2,156,424,064 | 229 |
| F8_E4M3 (block scales) | 1,905,738,240 | 5,968 |
| F32 (global scales) | 31,713,408 | 11,982 |

The index reconciles completely: 24,147 keys, key set identical to the headers,
every shard assignment agreeing, declared `total_size` 19,339,781,632.

Layer roles from real tensors: Mamba `[0, 2, 4, 7, 9, 11, …]` (23),
MoE `[1, 3, 6, 8, 10, 13, …]` (23), attention `[5, 12, 19, 26, 33, 42]` (6) —
identical to the `hybrid_override_pattern` and to N1's independently recorded
MoE layer list.

## 3. Layout adjudication — CONFIRMED

Tested field by field on **all 2,944 routed experts and all 23 shared experts**,
not on a sample. Every check passed for every expert.

For a routed expert (`hidden = 2688`, `moe_intermediate = 1856`, `relu2`):

| tensor | dtype | shape | bytes | meaning |
|---|---|---|---:|---|
| `up_proj.weight` | U8 | `[1856, 1344]` | 2,494,464 | 1344 = 2688/2, two 4-bit codes per byte |
| `up_proj.weight_scale` | F8_E4M3 | `[1856, 168]` | 311,808 | 168 = 2688/16, group-16 block scales |
| `up_proj.weight_scale_2` | F32 | scalar | 4 | global scale |
| `up_proj.input_scale` | F32 | scalar | 4 | activation scale |
| `down_proj.weight` | U8 | `[2688, 928]` | 2,494,464 | 928 = 1856/2 |
| `down_proj.weight_scale` | F8_E4M3 | `[2688, 116]` | 311,808 | 116 = 1856/16 |
| `down_proj.weight_scale_2` | F32 | scalar | 4 | |
| `down_proj.input_scale` | F32 | scalar | 4 | |
| **record** | | | **5,612,560** | |

Grouping runs along the **contraction dimension** in both matrices, which is the
kernel-friendly orientation: a group-16 block is contiguous in the direction the
GEMV reduces over.

Shared experts are the same structure at `intermediate = 3712`, giving
11,225,104 bytes each × 23 = 258,177,392 — the frozen N1 shared bucket.

**The N0R hypothesis is therefore confirmed, not merely consistent.** It was
tested against real dtypes, shapes and per-field byte counts, exactly as the
preregistration required, rather than against byte totals alone.

## 4. Random-access record layout — the H3-relevant result

safetensors groups tensors by dtype, so an expert's eight tensors are not one
run on disk. The question that matters is the *minimum number of contiguous
ranges* needed to fetch one expert.

**Every one of the 2,940 single-shard routed experts needs exactly three
contiguous ranges**, with no exceptions:

| class | runs | bytes |
|---|---:|---:|
| NVFP4 codes (both matrices) | 1 | 4,988,928 |
| FP8 block scales (both matrices) | 1 | 623,616 |
| FP32 globals (all four scalars) | 1 | 16 |

This is a materially better transport shape than a naive eight-range gather. It
holds because within each dtype region the two matrices of one expert are
adjacent (`down_proj` then `up_proj`, lexicographic), and the four FP32 scalars
of one expert are adjacent too.

**Four experts straddle a shard boundary** and need reads from two files:

| layer | expert | split |
|---:|---:|---|
| 8 | 34 | `up_proj.*` in shard 1, `down_proj.*` in shard 2 |
| 20 | 39 | `down_proj.weight` in shard 2, its scale/scalars in shard 3 |
| 31 | 63 | `up_proj.*` in shard 3, `down_proj.*` in shard 4 |
| 43 | 68 | `down_proj.weight` in shard 4, its scale/scalars in shard 5 |

Per the preregistration §6 this is **recorded, not treated as a defect**. It is
0.136% of experts and simply means the H3 bank builder must handle a two-file
gather for four records rather than assuming one file per expert.

## 5. Incompressible BF16 cost — an H4 input

229 BF16 tensors totalling **2,156,424,064 bytes (2.008 GiB)**, of which:

| item | bytes | note |
|---|---:|---|
| `lm_head.weight` | 704,643,072 | 131,072 × 2,688, BF16, excluded from quantization |
| `backbone.embeddings.weight` | 704,643,072 | same shape; `tie_word_embeddings = false`, so both are stored |
| everything else BF16 | 747,137,920 | six attention layers' q/k/v/o, six Mamba mixers' in/out_proj, all 23 `conv1d`, all norms, `A_log`, `D`, `dt_bias`, routers |

Embedding plus LM head alone are **1,409,286,144 bytes (1.312 GiB)** — 65% of
all BF16 bytes, on an 8 GiB budget. These two have opposite runtime profiles and
must be planned separately: the embedding is a one-row gather per token and is a
natural host resident, while the LM head is a full matvec every token. The
preregistered H4 ablation on host placement of the LM head or embedding is
therefore not optional bookkeeping — it is likely decisive, and it is now backed
by measured bytes rather than an estimate.

Confirming N0R's reading of `exclude_modules`: 5,968 U8 tensors = 2 × (2,944
routed + 23 shared + 17 trunk mixers). Seventeen is exactly the 23 Mamba layers
minus the six whose `in_proj`/`out_proj` are excluded.

## 6. Decoder validation — all four rules pass

Target: `backbone.layers.1.mixer.experts.0`, both matrices, **4,988,928 codes
each** (9,977,856 total), against a preregistered minimum of 1,048,576.

| rule | result |
|---|:--|
| 1 · range invariance — decoded elements lie in `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` | ✅ both matrices |
| 2 · bit-exact round trip — re-encode reproduces codes and packed bytes | ✅ both matrices |
| 3 · two independent implementations agree (E2M1, E4M3, and full dequant) | ✅ both matrices |
| 4 · structural — `N/2` code bytes, `N/16` scales, finite, no NaN | ✅ both matrices |

14 checks per matrix, 28 total, all true. The codec is implemented twice — a
table-driven path and a bit-arithmetic path that share no decode-time helper —
so agreement is evidence rather than a tautology. Its unit suite passes 16/16.

### Assumptions recorded, not proven

Two conventions are taken from the published format and are **not falsifiable by
self-consistency**:

1. **nibble order** `low_first` (element `2i` in the low nibble);
2. **dequantization grouping** `w = e2m1(code) × e4m3(weight_scale) × f32(weight_scale_2)`, with `input_scale` applying to activations rather than weights.

Both are flagged in `n2_decoder_validation.json` and must be confirmed in N3
against the official `modeling_nemotron_h.py` and one real forward. A wrong
nibble order would still pass every rule above while producing a wrong model —
which is exactly why this is recorded as an open assumption instead of a result.

## 7. Gates

Frozen preregistration §5 items this phase is responsible for:

| gate | result |
|---|:--:|
| g1 · five shards SHA-256-equal to frozen OIDs | ✅ |
| g2 · five headers parse, tensor count 24,147 | ✅ |
| g3 · tensor bytes 19,339,781,632 | ✅ |
| g4 · routed/shared/trunk buckets equal N1 | ✅ |
| g5 · layout hypothesis explicitly adjudicated | ✅ (**confirmed**) |
| g6 · decoder passes all four rules | ✅ |
| g7 · no BF16 materialization, no GPU, no timing | ✅ |
| g8 · artifacts ≤ 25 GiB | ✅ (18.014) |
| g9 · no protected byte changed | ✅ |

Gate 5 is written to pass on an explicit verdict either way; a falsified layout
would have been a valid scientific result. It happened to confirm.

Ten further observations were performed and reported per §3 but deliberately
kept out of the pass/fail set so they cannot alter the phase verdict. Nine are
true; the tenth — "every routed expert in a single shard" — is false by four
experts, which §6 designates as a recording, not a failure.

*Note on process:* the runner initially carried that contiguity check as a hard
gate, which the preregistration does not list. It was moved to observations to
match the frozen document. That is a correction of the runner toward the
preregistration, not a relaxation of a preregistered gate.

## 8. Claim boundary

Established: the local copy is byte-identical to the published checkpoint; the
exact on-disk tensor layout and declared quantization semantics; the true
routed/shared/trunk partition; the three-range random-access shape; and that one
routed expert decodes self-consistently under published NVFP4 semantics.

**Not** established: that these decoded values equal the publisher's BF16 source
weights (the BF16 checkpoint was deliberately not downloaded); model quality; any
latency or throughput figure; that the full model is correct; or that a runtime
is feasible. The 29.609 ms transfer floor remains a floor.

## 9. Known limitations

1. Nibble order and the `input_scale` role are unconfirmed conventions (§6).
2. Decoding was validated on one expert of 2,944; the layout was adjudicated on all of them, but only this one was decoded.
3. FP8 KV semantics are recorded as declared (`kv_cache_quant_algo: FP8`) and were not exercised; nothing here constrains runtime KV behavior.
4. The HF cache still holds a duplicate copy of the shards (~18 GiB, `.cache/nemotron_3_5_lightning/`). Disk is not constrained (256 GiB free) so it was left in place rather than deleted mid-phase; it is reclaimable at any time and is inside the Nemotron allowlist.

## 10. Nearest prior art

ModelOpt NVFP4 with FP8 group-16 block scales and FP32 global scales is a
published NVIDIA format; `nemotron_h` is published architecture with public
modeling code. Nothing in this phase is novel. Its value is that every number the
later phases depend on is now measured from the real artifact rather than
inherited.

## 11. Next falsification test

`N3_ONE_MODULE_REFERENCE`. The first job is to falsify the two open decoder
assumptions by comparing against the official `modeling_nemotron_h.py` path on
identical inputs. Then independent references for one Mamba-2 module, one GQA
attention module, the router with **one intercepted official top-6 call**, one
shared expert, one routed expert (`up → ReLU² → down`), FP8 KV write/read, and
one complete mixed module/residual path.

The router is the sharpest risk: the checkpoint carries
`gate.e_score_correction_bias` alongside `gate.weight`, with
`norm_topk_prob = true` and `routed_scaling_factor = 2.5`. Route selection is
therefore not a plain top-k over a linear projection, and the project's own
standing rule — never treat a recomputed `topk` as authoritative when the
official block returns IDs — applies directly.
