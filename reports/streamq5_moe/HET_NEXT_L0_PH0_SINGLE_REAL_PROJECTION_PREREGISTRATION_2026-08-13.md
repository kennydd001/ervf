# HET-NEXT-L0-PH0 single real projection — preregistration

Date: 2026-08-13  
State: **design frozen; implementation and physical execution closed**

## 1. Falsifiable question and claim boundary

Can one real, naturally selected Qwen3-Coder-Next layer-0 Q5 projection be evaluated by the same explicit width-8 ERGV reduction on the CPU oracle, Intel host-USM, and NVIDIA CUDA with exactly the same 512 BF16 output words?

A positive result proves only a **real-weight, real-activation, single-projection, sequential heterogeneous correctness component**. It does not prove a full expert, SwiGLU, routing, shared-expert, merge, layer, model, concurrent Intel+NVIDIA execution, throughput, latency, quality, capacity, or breakthrough claim. There is one already-used validation input and no held-out test arm.

No timing or performance statistic is admissible. Lifecycle timestamps may only diagnose hangs.

## 2. Frozen real identity

The projection is not chosen after observing device output. It is expert 50, route rank 0 of the official D2R3 `p0_n16` token-15 route and the first member of the earlier frozen Intel rank partition 0–3.

| Field | Frozen value |
|---|---|
| model revision | `a19358a7659bd1f564300250ee189120c49a562f` |
| shard | `model-00001-of-00040.safetensors` |
| shard bytes / SHA-256 | `3,999,619,288` / `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a` |
| safetensors header / data base | `194,000` / `194,008` bytes |
| tensor key | `model.layers.0.mlp.experts.50.gate_proj.weight` |
| dtype / shape / bytes | BF16 / `[512,2048]` / `2,097,152` |
| relative offsets | `[3,497,857,408, 3,499,954,560)` |
| absolute offsets | `[3,498,051,416, 3,500,148,568)` |
| source SHA-256 | `05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c` |
| Q5 codes SHA-256 | `20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b` |
| Q5 scales SHA-256 | `658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8` |
| codes+scales SHA-256 | `04e0e5591c051dd2c659a53263a4e2ac869d03f37daafb47d17e98ebdf924fa9` |
| decoded BF16 weight SHA-256 | `9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77` |

The route row, retained only as selection provenance, is `[50,199,237,474,245,374,239,8,168,12]`; its ten native BF16 selected-weight words are `[15999,15892,15878,15874,15782,15760,15723,15709,15683,15644]`. Routing weights do not enter this projection experiment.

The input is the zero-based token-15 row of D2R3 tensor `p0_whole_post_norm`, not a synthetic vector:

| Field | Frozen value |
|---|---|
| D2 raw | `reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors` |
| D2 raw bytes / SHA-256 | `171,696,126` / `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd` |
| full tensor | BF16 `[1,16,2048]`, absolute `[155,077,348,155,142,884)`, SHA `d82286fac9616cdf8b03b8eddb8347acd3679afb639c8db696daf3f643084853` |
| token-15 slice | BF16 `[2048]`, absolute `[155,138,788,155,142,884)`, 4,096 bytes |
| token-15 SHA-256 | `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f` |

Only this D2 range may be read. No other prompt, route, stage, tensor, model forward, or model weight is in scope.

## 3. Frozen STREAMQ5 record

Quantization is group-128, row-local. Source BF16 values widen exactly to FP32. For every group, `m=max(abs(w))` in FP32 reduction order, `s_fp32=m/15` if `m>0` else `1`, `q=clamp(round_ties_to_even(w/s_fp32),-15,15)` if nonzero else zero, stored field `q+15` in `[0,30]`, and stored scale `BF16(s_fp32)`. Eight little-order 5-bit fields occupy five bytes. Decode computes `BF16(FP32(q) * FP32(BF16 scale))`. Field 31 is forbidden.

The one in-memory wire record is exactly:

- header: 64 bytes, `struct <4sHHHBBIIH2xIII28s`;
- values: magic `SQ5M`, version 1, layer 0, expert 50, projection 0, bits 5, rows 512, columns 2048, group 128, code bytes 655,360, scale bytes 16,384, CRC32, and 28 zero reserved bytes;
- codes: offsets `[64,655,424)`;
- BF16 little-endian scales: `[655,424,671,808)`;
- zero padding: `[671,808,675,840)` (4,032 bytes);
- CRC32: `zlib.crc32(scales, zlib.crc32(codes)) & 0xffffffff`;
- total: **675,840 bytes**.

Before any device API is loaded, the CPU build phase must reproduce all four existing source/codes/scales/decoded digests above, scan all 1,048,576 fields with no 31, verify the header/CRC/padding, and freeze the derived header, CRC, record SHA, and CPU-oracle output SHA in a create-new input commitment. A mismatch is a negative result and forbids device submission.

## 4. Exact width-8 arithmetic contract

For every output row, `packs=256`, width 8, virtual slots 32. Lane `l` owns packs `l+8v` for `v=0..31`. Each pack decodes columns `8*pack..8*pack+7`; scale index is `row*16 + floor(column/128)`. Each weight is rounded to BF16 and widened to FP32; each input BF16 word is widened to FP32. The within-pack accumulator starts at positive FP32 zero and applies eight IEEE-754 binary32 fused multiply-add operations in increasing column order.

The 32 per-lane partials reduce with explicit FP32 round-after-add at virtual strides 16, 8, 4, 2, 1. The eight lanes then reduce by subgroup shuffle-down offsets 4, 2, 1, also with explicit FP32 round-after-add. Lane 0 rounds once, ties-to-even, to the retained BF16 output word. NaNs, infinities, compiler fast-math, reassociation, contraction of the add tree, FTZ/DAZ, or a different reduction order fail the contract.

The independent CPU oracle must implement this DAG from the wire bytes without importing either device kernel or the STREAMQ5 builder. A frozen software IEEE-binary32 FMA/add oracle is normative; a native `std::fma` implementation may only be used after bitwise agreement on edge-case fixtures.

The physical launch is frozen as **16 work-groups/blocks × 256 work-items/threads**. Each group contains 32 logical width-8 rows; `row = group_id*32 + subgroup_id`. Thus every row 0–511 is written exactly once and no other row is addressable.

## 5. Sequential physical arms

Order is fixed: CPU build/oracle → negative controls → Intel arm and complete cleanup → NVIDIA arm and complete cleanup → independent CPU verifier. Intel and NVIDIA must never be active concurrently.

### Intel

- exact Intel Arc device and required subgroup-8 plus Intel USM capability are hard gates;
- three host-USM allocations: record 675,840 bytes, input 4,096 bytes, output 1,024 bytes (680,960 total);
- CPU initializes record/input directly through their USM pointers and reads output only after kernel event completion and queue finish;
- only `clSetKernelArgMemPointerINTEL` binds these buffers;
- `clCreateBuffer`, `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`, `clEnqueueCopy*`, and migration APIs are forbidden and their call counters must stay zero;
- `clGetMemAllocInfoINTEL` must prove pointer base, host allocation type, and exact size for all three allocations;
- the OpenCL source is an exact 675,840-byte/512×2048 adaptation of the already evidenced width-8 reduction, with no fast-relaxed-math. Build log, source SHA, binary SHA, device/driver/extensions, enqueue count one, and all release attempts are retained.

### NVIDIA

- exact NVIDIA device identity, CUDA driver/runtime, compute capability, and sufficient free memory are hard gates;
- pinned host allocations: record 675,840, input 4,096, output 1,024 bytes; device allocations with the same three sizes;
- exactly two H2D copies (record and input), one width-8 kernel launch on a nondefault stream, and one D2H copy (output); no dense/dequantized weight allocation;
- input and record pointers are distinct kernel parameters; kernel mapping and arithmetic are identical to the CPU contract;
- raw source/PTX or CUBIN bytes, compile log/options, source/binary SHA, allocation/copy/launch ledger, output, synchronize status, and every cleanup attempt are retained.

## 6. Controls and gates

All controls run before any valid device submission; their global Intel and NVIDIA enqueue/launch counters must remain zero.

1. **Field 31:** set row 0, pack 0, slot 0 to field 31, recompute CRC so structural CRC passes, and require rejection specifically by the exhaustive field-range scan.
2. **Wrong identity:** change the header expert from 50 to 51 and require rejection by requested-versus-presented identity before dispatch.
3. **Wrong digest:** flip bit 0 of code byte 5 without updating CRC/digests and require CRC and canonical-payload rejection.

The final positive conjunction is exact and non-compensating:

- all frozen file/range/source/codec commitments match;
- CPU, Intel, and NVIDIA each retain exactly 512 finite BF16 words / 1,024 raw bytes;
- `CPU words == Intel words == NVIDIA words` bitwise, with 0 differing words and equal SHA-256;
- exactly 512 unique row writes and no sentinel remains;
- all three negative controls reject at the predeclared stage with zero device submissions;
- Intel forbidden-copy counters are zero; NVIDIA copy/launch counts are exactly 2/1/1;
- each device is fully synchronized and all owned resources have one attempted, successful release; Intel is closed before NVIDIA initialization;
- process peak working set ≤2 GiB, start available RAM ≥2 GiB, NVIDIA device allocation total exactly 680,960 bytes and start free VRAM ≥64 MiB;
- create-new raw/result/verifier artifacts are finite, self-hashed, atomically committed, independently verified, and total new persistent evidence ≤4 MiB. The 675,840-byte record remains in RAM and is not a persistent bank.

Any mismatch, missing evidence, device/runtime incompatibility, resource/lifecycle error, or verifier error is a valid negative or blocked result—not permission to retune, reorder, retry, or relax bitwise equality.

## 7. Locked prior evidence

- selected-source manifest: `reports/streamq5_moe/het_next_l0_pv0r2_selected_source_manifest.json`, SHA `0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`;
- D2 result: SHA `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`;
- D2 independent artifact audit: SHA `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`;
- D2 interpretation: SHA `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`;
- S0-R5 result/raw: SHA `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91` / `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`;
- codec contract source: `scripts/streamq5_moe/port80b_t0q5r3_codec_contract.py`, SHA `f78ce418b284d414091564e3667f9989f3556c72892306536f96aebbd38360b0`;
- Intel width-8 evidence source/result: SHA `6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca` / `f4f1763606c925001f62504511300ca4bb7728329650e63604eb2921c70c79b2`;
- NVIDIA ERVF source: `scripts/streamq5_moe/run_p7b_ervf_kernel.py`, SHA `2248c05b9a8da3b9ab58dab779dabeb5fe1c453c04626230d0c6eeca1f62cba8`.

No prior artifact by itself authorizes PH0 implementation or execution.
