# HET-NEXT-L0-PH0-R1 single real projection — preregistration

Date: 2026-08-13  
State: **immutable design; implementation, preflight and device execution closed**

This is the complete superseding PH0 protocol. It binds the immutable PH0 draft (SHA `1c2a202d5d90896233d9f88452c371b07aa429cbb08b9e28e03e86d61ad0990a`) as provenance, but no conflicting clause from that draft remains normative.

## Claim and one real case

Question: does one official naturally selected Qwen3-Coder-Next Q5 projection produce the same 512 BF16 words under one explicit width-8 reduction on an independent CPU oracle, Intel host-USM, and NVIDIA CUDA, executed sequentially?

A positive outcome is only a validation-only, real-weight/real-activation, single-projection heterogeneous correctness component. It is not a full expert, routing, shared, merge, layer, model, concurrency, quality, capacity, latency, throughput, or breakthrough result. No performance metric is admissible.

The fixed projection is route-rank-0 expert 50 gate projection from D2R3 `p0_n16` token 15:

- official revision `a19358a7659bd1f564300250ee189120c49a562f`, shard 1 size `3,999,619,288`, SHA `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`;
- key `model.layers.0.mlp.experts.50.gate_proj.weight`, BF16 `[512,2048]`, absolute byte range `[3,498,051,416,3,500,148,568)`, source SHA `05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c`;
- expected Q5 codes/scales/combined/decoded SHA: `20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b`, `658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8`, `04e0e5591c051dd2c659a53263a4e2ac869d03f37daafb47d17e98ebdf924fa9`, `9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77`;
- input is only D2R3 `p0_whole_post_norm` token-15 BF16 `[2048]`, absolute `[155,138,788,155,142,884)`, 4,096 bytes, SHA `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f` from raw SHA `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`.

No other shard or D2 payload range is allowed.

## Exact Q5 wire and CPU commitment

The group-128 codec widens source BF16 to FP32, uses FP32 `max(abs(w))/15`, `torch.round` ties-to-even, clamp `[-15,15]`, zero groups as q=0 and scale BF16(1), stores `q+15` in `[0,30]`, and packs eight fields little-order into five bytes. Decode is `BF16(FP32(q)*FP32(BF16 scale))`. Field 31 is invalid.

The sole in-memory record is 675,840 bytes: 64-byte header `<4sHHHBBIIH2xIII28s`, 655,360 code bytes, 16,384 BF16 scale bytes, and 4,032 zero padding bytes. Header values are `SQ5M,1,layer0,expert50,projection0,bits5,rows512,cols2048,group128,655360,16384,crc32,zeros28`; CRC is `zlib.crc32(scales,zlib.crc32(codes)) & 0xffffffff`.

Expected pristine derived evidence is fixed:

- CRC32 `1,976,639,022` (`0x75d11e2e`);
- header SHA `7ce36f740b0348434aeff2ee58b1b656a9480d4553281336a09b87f3c653f699`;
- record SHA `e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9`;
- first 12 code bytes `d74fe56065329ab65c6532cd`; first scale word `0x3b14`.

Before a device library is loaded, the CPU build must independently reproduce every identity/digest, scan all 1,048,576 fields, retain q extrema/zero groups and padding, and commit the independent explicit CPU-oracle output.

## Exact width-8 oracle and launch

For each row: 256 packs, eight lanes, 32 virtual packs per lane. Lane `l` consumes pack `l+8v` (`v=0..31`), fields and columns in increasing order, with eight IEEE binary32 fused multiply-adds from positive zero. Every decoded weight is rounded to BF16 then widened; each input BF16 word is widened. Per-lane virtual partials reduce with explicit FP32 round-after-add at distances 16,8,4,2,1; the eight lanes reduce at offsets 4,2,1. Lane zero performs one ties-to-even BF16 output round.

The normative CPU implementation is a separate software IEEE-binary32 FMA/add DAG; native `fmaf` is only a cross-check. Fast math, FTZ/DAZ, reassociation, altered contraction, NaN or infinity fail.

Both device launches are exactly 16 blocks/groups ×256 threads/items. Each block has 32 logical width-8 rows: `row=block*32+floor(thread/8)`, `lane=thread%8`. Exactly one uint32 counter per row is incremented after its output write.

CUDA must use `cooperative_groups::tiled_partition<8>(cooperative_groups::this_thread_block())` and tile-local `shfl_down` offsets 4,2,1 (or an independently audited `__shfl_down_sync(__activemask(),value,offset,8)` exact equivalent). Full-warp shuffles without width 8 fail. Field validity is an exhaustive unpacked-field `<=30` check; source/static audit rejects any five-byte/`0xff` sentinel heuristic as a substitute.

## Frozen device identity policy

Exactly one matching Intel and one matching NVIDIA device must exist and their PCI addresses must differ:

- Intel: `Intel(R) Arc(TM) Pro 140T GPU (32GB)`, vendor/device/subsystem `8086:7d51:2346:17aa`, PNP revision `03`, PCI `0000:00:02.0`, driver family `32.0.101.x` with observed version at least `32.0.101.8517` and below `33.0.0.0`;
- NVIDIA: `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, vendor/device/subsystem `10de:2d39:2346:17aa`, PNP revision `A1`, PCI `0000:01:00.0`, NVIDIA display driver at least `595.71` and below `600.00` (Windows observed encoding must be recorded exactly), exact CUDA device 0 only.

Name, full PCI/PNP identity, driver/runtime/compiler versions, extension/capability bits, and enumeration cardinality are retained. A different or ambiguous device is `blocked_capability`; it is never auto-selected.

## Device arms and exact byte/call ledger

Order is CPU/controls → Intel → complete Intel cleanup → NVIDIA → complete NVIDIA cleanup → independent verifier. No overlap.

Intel semantic host-USM is record 675,840 + input 4,096 + output 1,024 = **680,960 bytes**. Diagnostic host-USM is uint32 counters `[512]` = **2,048 bytes**; total Intel host-USM is **683,008 bytes**. CPU writes record/input and explicitly zeroes counters/output through their host pointers. Kernel binds all pointers only with `clSetKernelArgMemPointerINTEL`; CPU reads only after event completion and `clFinish`. `cl_mem`, OpenCL write/read/copy/map/unmap/migrate calls are forbidden (zero calls). `clGetMemAllocInfoINTEL` proves host type/base/exact sizes.

NVIDIA pinned host allocations are record 675,840, input 4,096, output 1,024, counters 2,048 = **683,008 bytes**. Device allocations are the same = **683,008 bytes**; combined pinned+device is **1,366,016 bytes**. Exact command ledger on one nondefault stream:

1. `cudaMemsetAsync(device_output,0xff,1024,stream)` diagnostic sentinel;
2. `cudaMemsetAsync(device_counters,0,2048,stream)`;
3. H2D record 675,840;
4. H2D input 4,096;
5. one 16×256 kernel launch;
6. D2H output 1,024;
7. D2H counters 2,048;
8. one stream synchronize before host inspection.

Thus semantic H2D count is 2, D2H count is 2 (one semantic output plus one diagnostic counter), kernel count is 1, memset count is 2. No dense/dequantized weight exists. Raw source/PTX-or-CUBIN, compile log/options, allocation identities, call offsets/sizes, return codes, outputs/counters, sync, and all cleanup attempts are retained.

## Pre-device negative and sensitivity controls

Every safe-parser control runs before device initialization and must leave Intel enqueue and NVIDIA launch counters at zero. Parser order is exact: size → header/schema/identity → CRC → canonical source/codes/scales/input digests → exhaustive field scan → dispatch.

1. **Truncation:** pristine record minus final byte (`675,839`) rejects at size.
2. **Wrong identity:** expert header 50→51 rejects at identity.
3. **CRC-invalid code:** code byte index 5 `0x32→0x33`, stale CRC/digests, rejects at CRC.
4. **Code digest after valid CRC:** the same byte mutation with recomputed CRC `2,106,093,510` passes CRC but rejects canonical code digest; mutated codes SHA `0f980e086029b664b4d7705349bb184732f35ccd22707623cb8aa49bcd59fdd9`, record SHA `61e77ea30b82021ae92baf0ace92906b482c0e94088b271c0de941a94013217e`.
5. **Wrong scale:** scale word 0 `0x3b14→0x3b15`, recomputed CRC `3,896,522,641`, rejects canonical scale digest; mutated scale SHA `df46052d03f183db7029478f0e715b21862378e837d30a6f3b1a572810a907a5`.
6. **Wrong input:** input word 0 `0xbe34→0xbe35` rejects the frozen input SHA before record dispatch.
7. **Field 31:** row0/pack0/slot0 field `23→31`, first pack becomes `df4fe56065`; recomputed CRC `3,350,049,836` passes, canonical digests for this test are explicitly presented, then exhaustive field scan rejects specifically field31. Mutated record SHA `b7cd621738866a88f75ef1c9c70ead443f9b5bbe32cd9ca3147b716bc3558701`.
8. **Deterministic one-step sensitivity:** row0/column1 q `15→14` (stored field `30→29`), first pack `d74fe56065→b74fe56065`, all other fields/scales unchanged, CRC recomputed to `1,107,492,658`. Safe checker rejects canonical codes digest (mutated codes SHA `9543240c319a1708716f986433ee2f5638bde5f91ad99373de7b349588478619`). An explicitly labelled CPU-only unsafe bypass must change row0 output BF16 word from `0xbe53` to `0xbe52`; mutated record SHA is `9cf72228b4215f3637c2c1a89f4bc270a8bdbbd6636c5bbd22733b16639059a3`. No device sees this mutation.

## Hard positive conjunction

- All frozen bindings, identities, record evidence, parser-order controls, and sensitivity evidence pass.
- CPU, Intel and NVIDIA each retain exactly 512 finite BF16/uint16 words and 512 counters all equal 1; output/counter raw bytes and SHA are retained.
- CPU words equal Intel words equal NVIDIA words bitwise: 0 differing words and equal SHA.
- Intel copy-family forbidden calls are zero; NVIDIA exact calls are memset2/H2D2/kernel1/D2H2/sync1.
- Intel is fully released before NVIDIA initialization; every owned release is attempted exactly once and succeeds.
- Start available RAM ≥2 GiB, peak working set ≤2 GiB, NVIDIA start free VRAM ≥64 MiB, exact device bytes 683,008, exact pinned bytes 683,008, Intel host-USM 683,008.
- Atomic create-new raw/result/verifier evidence totals ≤4 MiB and is independently verified. No persistent Q5 record/bank is created.

Any false conjunct is negative/blocked/valid-failure. No retry, retune, changed mutation, changed reduction, or tolerance replaces the bitwise gate.

## Bound small evidence

The source manifest SHA is `0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`; D2 result/audit SHA are `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb` and `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`; S0-R5 result/raw SHA are `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91` and `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`; hardware baseline SHA is `4d80a6fc54ef1c7432bf6c897041bd6f3452499349cacfbf23ae5b5896e0b3da`; CAP0X-R2 result/audit SHA are `d807e0867c41ba43e0ee86b2bbf6d14bba7db582d7084db390472739714f2a3d` and `4f827ae8e96acbe6dd2df6aff377ee585e270796f16fb9b9e68308907f09b058`.
