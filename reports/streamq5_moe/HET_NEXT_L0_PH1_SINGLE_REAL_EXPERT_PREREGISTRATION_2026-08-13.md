# HET-NEXT-L0-PH1 — one complete real expert on CPU, Intel and NVIDIA

Date: 2026-08-13  
State at freeze: **design only / execution closed**. No PH1 payload read, compiler call, device enumeration, allocation, kernel launch, or output exists. This document authorizes implementation only after an independent design audit; implemented sources require a second audit before any static preflight or physical action.

## Question and claim boundary

PH0X-R12 independently proved one official Qwen3-Coder-Next layer-0 Q5 gate projection bit-for-bit on CPU, Intel Arc host-USM, and NVIDIA cubin. PH1 asks the next conjunctive question: can the same two physical GPUs execute the complete unweighted expert-50 path

`post_norm[15] -> gate_proj + up_proj -> deterministic BF16 SiLU -> BF16 activation -> down_proj`

using three real official Q5 records, with every device-produced stage matching an independent bit-level CPU oracle?

A PH1 positive supports only one known, naturally routed, real expert on one known natural activation. It is validation evidence, not held-out evidence. It is not a top-10 composition, shared expert, layer, full-model, concurrency, timing, throughput, deployment, novelty, industrial-readiness, or breakthrough claim. No routing weight is applied in PH1.

## Immutable upstream evidence

- Official checkpoint: `Qwen/Qwen3-Coder-Next`, revision `a19358a7659bd1f564300250ee189120c49a562f`.
- Shard: `model-00001-of-00040.safetensors`, exactly `3,999,619,288` bytes, SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- D2-R3 raw: `t0r12d2_raw.safetensors`, exactly `171,696,126` bytes, SHA-256 `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`.
- Natural input: D2 key `p0_whole_post_norm`, token index `15`, absolute range `[155138788,155142884)`, BF16 `[2048]`, exactly `4,096` bytes, SHA-256 `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f`.
- Natural route IDs at that token are exactly `[50,199,237,474,245,374,239,8,168,12]`, raw I64 SHA-256 `ea47c4b4b3b2942876101be4dc85072554805de8fef20d91ab531b64c731a462`. Expert 50 is official rank 0. The ten BF16 route-weight words are `[15999,15892,15878,15874,15782,15760,15723,15709,15683,15644]`, SHA-256 `249c79806e09cf86b0bd6aba465050621dea3163d362fffea5ace2f655e7c8a7`.
- Selected-source manifest: `het_next_l0_pv0r2_selected_source_manifest.json`, exactly `22,287` bytes, SHA-256 `0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`.
- PH0X-R12 result SHA-256 `159d77b8fc6d1cac3d2123c09b7f256837c9691a397fa9d052309752d26955bc`.
- PH0X-R12 independent verification JSON SHA-256 `7ee3161f03b79be6475fec9ddc4936a28019632640e81b02ed18ed3b132e1b9b`; independent verifier SHA-256 `caae0492d274d961fee53af85971a18d40e6e391b7db17c816a1eafed0e681`; report SHA-256 `6904cbc34eb935e55e455c8f006e75d0dbc6b1c46247a5c3fbb445a4001e344a`.

Only the three official source ranges below and the one D2 input range may be read. Generic safetensors loading, full-payload hashing during execution, model construction, and any p1-p3 D2 value access are forbidden.

## Exact real records

Each source is BF16 and exactly `2,097,152` bytes. Quantization is symmetric group-128: source BF16 to FP32; `scale=max(abs(group))/15` in FP32; an all-zero group uses BF16 scale `1` and q=0; otherwise round-to-nearest ties-to-even; clamp q to `[-15,15]`; store `q+15` in `[0,30]`; field 31 is forbidden; pack eight little-order five-bit fields per five bytes; cast scale once to BF16. Decode is FP32(q) times FP32(BF16 scale), cast once to BF16 RNE.

The 64-byte record header is the frozen PH0 format `<4sHHHBBIIH2xIII28s>` with magic `SQ5M`, version 1, layer 0, expert 50, projection ordinal 0/1/2, codec 5, rows, columns, group 128, code bytes `655,360`, scale bytes `16,384`, chained IEEE CRC32 over codes then scales, and 28 zero reserved bytes. Record layout is header64 + codes655360 + scales16384 + zero padding4032 = `675,840` bytes. The triplet is `2,027,520` bytes.

| projection | ordinal | shape | absolute source range | source SHA-256 | codes SHA-256 | scales SHA-256 | decoded SHA-256 | CRC32 | record SHA-256 |
|---|---:|---:|---:|---|---|---|---|---:|---|
| gate | 0 | `[512,2048]` | `[3498051416,3500148568)` | `05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c` | `20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b` | `658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8` | `9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77` | `1976639022` | `e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9` |
| up | 1 | `[512,2048]` | `[3500148568,3502245720)` | `4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d` | `6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb` | `c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9` | `ca239543f7a478e757040a994d001a15b70481c7b87bca3cc8641831305394ea` | `4920057` | `6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb` |
| down | 2 | `[2048,512]` | `[3495954264,3498051416)` | `bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479` | `3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703` | `a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e` | `ef9c19383d9b1ff90a4ba0015942594c4188dd42c407103a06f26a1953d56c34` | `4066311128` | `bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157` |

The builder and independent verifier must separately reread, rehash, requantize, reconstruct every record byte, reject field31, and reproduce all listed hashes before a device may open.

## Deterministic arithmetic contract

### Q5 linear

For an `R x C` projection (`512x2048` for gate/up; `2048x512` for down), each output row has eight logical lanes. Lane `l` owns pack `t=l+8*v`; `v=0..31` for `C=2048` and `v=0..7` for `C=512`. A pack starts FP32 +0. For fields `i=0..7` in order, decode the BF16 weight endpoint, widen weight and BF16 input to FP32, and perform one IEEE-754 FP32 fused multiply-add RNE. Reduce the per-lane pack partials with the exact balanced tree (`16,8,4,2,1` for gate/up; `4,2,1` for down), then reduce eight lane values with offsets `4,2,1`; cast once to BF16 RNE. FTZ/DAZ are forbidden. No reassociation, fast math, native dot primitive, Torch/NumPy sum, or tolerance is allowed.

Launches are fixed at block/local size 256. Gate and up use grid 16 (512 rows); down uses grid 64 (2048 rows). Each row counter is uint32, starts zero, and must finish exactly one; every BF16 output starts `0xffff` and must be overwritten.

### Exact BF16 SiLU and activation

To avoid backend-specific `exp` and FTZ behavior while retaining the official BF16 function, PH1 freezes a complete 65,536-entry table indexed by the raw BF16 gate word. It is generated once under CPU PyTorch `2.12.1+cu132`, one thread, deterministic algorithms, flush-denormal false: enumerate words 0..65535; for each finite BF16 word store the raw BF16 word returned by `torch.nn.functional.silu(input_bf16, inplace=False)`; for the 256 NaN/Inf encodings store zero. The table is exactly `131,072` little-endian bytes, has `65,280` finite mappings, and SHA-256 `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed`. Gate output must be finite before lookup, so a nonfinite gate fails rather than consuming a sentinel entry.

The device activation kernel reads its own gate and up words, obtains `silu_word = LUT[gate_word]`, and computes `activation_word = BF16_RNE(exact(FP32(silu_word) * FP32(up_word)))` using a specified integer/bit-level IEEE routine, not native floating multiplication. Signed zero is retained; NaN/Inf input or output fails. Thus gate, up, SiLU and activation are bitwise portable. Its 512 uint32 row counters must all finish one.

The down kernel consumes the device-produced activation buffer directly. Host substitution or rewrite of gate, up, SiLU, activation, or down between enqueue/launch and final synchronization is forbidden.

## Source-quality and device gates

The independent CPU source arm rereads all three official BF16 matrices and exactly reproduces the official graph: concatenate gate then up weights into BF16 `[1024,2048]`, execute one PyTorch BF16 `F.linear(input, gate_up)`, split its `[1024]` output into gate then up `[512]`, apply `F.silu(gate, inplace=False)`, native BF16 `silu*up`, then `F.linear(activation, down)`. Separate source gate/up calls are forbidden because their output shape may select a different BF16 reduction implementation. This individual expert is also bound to the immutable S0-R5 evidence in which the complete selected source graph reproduced the D2 routed aggregate.

The independent CPU Q5 arm uses the exact Q5 linear DAG, LUT, and integer BF16 multiply above. Before any device opens:

1. every source/record/input/LUT/provenance gate passes;
2. every raw source and Q5 tensor is finite;
3. CPU-Q5 down versus source-BF16 down has FP64 row-major `rel_l2 <= 0.08`; no percentile or retuning;
4. all safe rejection and numerical-sensitivity controls pass.

For each physical device, all 512 gate words, 512 up words, 512 SiLU words, 512 activation words, and 2048 down words must equal the independent CPU-Q5 words bit-for-bit. All counters, canaries, exact buffer/copy/enqueue ledgers, identities, and release gates are conjunctive. Intel must pass and cleanly release before NVIDIA may open. A negative never opens a later phase.

## Controls

The checker order is fixed: size -> structural header without requested tuple comparison -> CRC -> field scan -> official source plus pristine code/scale digests -> requested tuple -> input digest -> full-record digest -> dispatch. The LUT digest is checked after input and before dispatch. For each of the three records, the static preflight and independent verifier reconstruct exact failures for truncation, wrong projection ordinal, stale CRC, valid-CRC code mutation, scale mutation, recomputed-CRC field31, wrong input and wrong LUT. No compile/allocation/launch counter may increment on rejection.

For each projection, select the lexicographically first nonzero q using pristine codes/scales only, move it one step toward zero, mutate exactly one packed field with unchanged scale/all other fields, and require the safe checker to reject the altered code digest. With a one-hot BF16 `2^-8` input at its source column, an unsafe CPU diagnostic must alter exactly the selected output row and reproduce this frozen evidence:

| projection | selector `(row,col,group,pack,slot)` | q -> q' | pack before -> after | mutated codes SHA-256 | activation SHA-256 | output word before -> after | original / mutated output SHA-256 |
|---|---|---|---|---|---|---|---|
| gate | `(0,0,0,0,0)` | `8 -> 7` | `d74fe56065 -> d64fe56065` | `86f2470ba33598306b388cb2762686e3cd5b140e90eff5289606003fbd83b224` | `2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561` | `0x3894 -> 0x3882` | `98fac647d0adc50536d5b397b1974ac237ec14a818ac4ec287760dbab312400b` / `3571cfc8dbc22de68d5b216fa5766b3bc0036e745062dda3d645fc1b1c019910` |
| up | `(0,0,0,0,0)` | `-6 -> -5` | `09c0195da0 -> 0ac0195da0` | `e92f4a5ce9a73e1b5e0cadbb3792b2ab150cd0ffb58e46ffd5f2471f8f59863f` | `2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561` | `0xb85a -> 0xb835` | `e6e969c95a005aebb925300e1771a20d80b026366cb7287b53f381d5cb4846c8` / `17f7ef7778ce24587a8b53efccedfb76ba0717fd350917ff4b86c269ca8a5069` |
| down | `(0,0,0,0,0)` | `-7 -> -6` | `28aaf71e78 -> 29aaf71e78` | `ed864ba58860f321ccdd141d1333a021f7eb8954499bc0bbcb93665cf6684e64` | `080988a1eaa25baa23e23e7299ef6a97ba1fb8c682fd05925b0e4c7a3543b153` | `0xb8aa -> 0xb892` | `063ad04f9b3fe6d96bbbd874ae7ff685d8311a792bcb76495e95ab303a55f43d` / `c0a7e675d316faee66922666eae9508e897f5a3a41b24b36820791a421770286` |

The natural outputs are never searched to choose these controls.

## Physical contracts and resources

Intel target is exactly the PH0X-proven `Intel(R) Arc(TM) Pro 140T GPU (32GB)`, vendor Intel, driver `32.0.101.8517`, PCI `0000:00:02.0`, with `cl_intel_unified_shared_memory`. Every record, input, LUT, stage output and counter is a separate 4096-aligned host-USM allocation and every kernel pointer is bound with `clSetKernelArgMemPointerINTEL`. No `cl_mem`, CreateBuffer, enqueue read/write/copy, migrate, prefetch, or explicit device-copy call is allowed. Exact host-USM semantic bytes are `2,185,216`; no unlisted allocation. Four in-order enqueues (gate, up, activation, down), then one `clFinish`; CPU reads only after finish.

NVIDIA target is exactly `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, driver/runtime `13020`, PCI `0000:01:00.0`. Direct precompiled no-FTZ cubins are required; no CuPy/NVRTC source compile and no textual PTX path. Use one nondefault stream. Pinned-host and device tables each contain the same named allocations totaling `2,185,216` bytes, combined `4,370,432`; no pool or implicit pageable copies. Exact calls: H2D for three records, input and LUT (5); memset for five stage outputs and four counter arrays (9); four kernels; D2H for five stages and four counters (9); one stream synchronization; reverse release of every device allocation, pinned allocation and stream despite earlier release errors. Compiler/build diagnostics and cubin bytes/hashes are retained.

Start available host RAM must be at least 16 GiB; each runtime sample must leave at least 2 GiB available and process peak working set must not exceed 12 GiB. Intel bytes must equal `2,185,216`; NVIDIA combined pinned+device bytes must equal `4,370,432`. Final retained PH1 artifacts must be at most 16 MiB. There is no timing gate.

## Evidence, lifecycle and outcomes

Retain canonical package bytes, the LUT, raw CPU-source/CPU-Q5/Intel/NVIDIA stage words, counters/canaries, source/record evidence, metrics, identities, compiler artifacts, exact call/allocation/release ledgers, resource samples and an exact raw manifest. An independent verifier may import no runner, builder, codec, oracle or backend module; it rereads the allowlisted source/input bytes, independently rebuilds all records/LUT/oracles/metrics and checks every physical conjunct. It also mutates each positive conjunct (one raw word, record byte, LUT byte, counter, call byte/direction, device identity and cleanup row) and requires rejection.

The run directory must be absent before authorization. Use create-new temp -> write -> fsync -> independent precommit check -> rename -> result -> commit-last, with bounded recovery/quarantine and immutable failure evidence. A valid commit is never modified. One authorized physical attempt only; no retune or retry.

Outcomes are `positive`, `negative_source_or_quality`, `negative_intel`, `negative_nvidia`, `negative_control_or_protocol`, `blocked_capability`, `blocked_resource`, or `invalid_execution`. A PH1 positive requires every conjunct above; otherwise the narrow claim remains unopened.
