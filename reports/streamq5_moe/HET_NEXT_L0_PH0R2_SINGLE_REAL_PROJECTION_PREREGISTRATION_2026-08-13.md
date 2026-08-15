# HET-NEXT-L0-PH0-R2 single real projection — preregistration

Date: 2026-08-13  
State: **final immutable design; implementation, preflight and execution closed**

This supersedes PH0-R1 (preregistration SHA `f4693c42020f47ce2f7d09b655918fb6fb63e93134d35a0fe1af353415b9734f`; design SHA `68e531219419fd39048d831ab3c078daf5e2d5e6f90b312a767c87950f6828c2`). R1 remains immutable evidence. R2 changes only control selection/checker order and explicit sentinel initialization; all scientific scope, real input, real weight, exact arithmetic, devices, resources and claim limits stay unchanged.

## Claim and immutable real payload

Question: does one naturally selected official Qwen3-Coder-Next Q5 projection yield the same 512 BF16 words under one explicit width-8 reduction on an independent CPU oracle, Intel host-USM, and NVIDIA CUDA, run sequentially?

A positive outcome proves only a validation-only, real-weight/real-activation, single-projection heterogeneous correctness component. It proves no full expert, routing, shared, merge, layer, model, concurrency, quality, capacity, timing, throughput or breakthrough claim. No performance measurement is allowed.

The fixed matrix is route-rank-0 expert 50 gate projection:

- revision `a19358a7659bd1f564300250ee189120c49a562f`; shard-1 bytes `3,999,619,288`, SHA `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`;
- key `model.layers.0.mlp.experts.50.gate_proj.weight`, BF16 `[512,2048]`, absolute `[3,498,051,416,3,500,148,568)`, source SHA `05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c`;
- pristine Q5 codes/scales/combined/decoded SHA: `20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b`, `658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8`, `04e0e5591c051dd2c659a53263a4e2ac869d03f37daafb47d17e98ebdf924fa9`, `9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77`.

The natural physical input is exclusively D2R3 `p0_whole_post_norm` token 15, BF16 `[2048]`, absolute `[155,138,788,155,142,884)`, SHA `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f`, from raw SHA `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`. No other shard or D2 payload range is readable.

## Pristine codec and record

Group-128 codec: widen BF16 source to FP32; FP32 `max(abs(w))/15`; `torch.round` ties-to-even; clamp `[-15,15]`; zero-group q=0 and BF16 scale 1; stored field `q+15` in `[0,30]`; eight little-order fields/five bytes; decode `BF16(FP32(q)*FP32(BF16 scale))`. Field31 is forbidden.

The only in-memory wire is 675,840 bytes: header64 `<4sHHHBBIIH2xIII28s`, codes655,360, BF16 scales16,384, zero padding4,032. Header is `SQ5M,1,layer0,expert50,projection0,bits5,rows512,cols2048,group128,code655360,scale16384,CRC,zeros28`. CRC is `zlib.crc32(scales,zlib.crc32(codes))&0xffffffff`.

Pristine commitments: CRC `1,976,639,022`; header SHA `7ce36f740b0348434aeff2ee58b1b656a9480d4553281336a09b87f3c653f699`; record SHA `e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9`; first code pack `d74fe56065`; first scale word `0x3b14`.

Before device initialization, the CPU phase reproduces all source/codec/record identities, exhaustively scans all 1,048,576 fields, and commits the independent CPU-oracle output.

## Exact width-8 arithmetic

For each row: 256 packs; lane `l` owns `l+8v`, `v=0..31`; eight increasing-column IEEE binary32 fused multiply-adds from positive zero per pack; decoded weight BF16→FP32 and input BF16→FP32. Virtual partials reduce with FP32 round-after-add distances 16,8,4,2,1; eight lanes reduce 4,2,1; lane0 does one BF16 ties-to-even round. The normative CPU oracle is an independent software IEEE binary32 DAG. Fast math, FTZ/DAZ, reassociation, changed contraction, NaN or infinity fail.

Each device launches exactly 16×256. `row=block*32+floor(thread/8)`, `lane=thread%8`. OpenCL requires subgroup8. CUDA requires `cooperative_groups::tiled_partition<8>` and tile `shfl_down` 4,2,1, or independently audited `__shfl_down_sync(__activemask(),...,width=8)`. Full-warp unbounded shuffles fail. Each of 512 uint32 row counters increments exactly once after output write.

## Exact devices and sequential resources

Exactly one match per backend, with distinct PCI:

- Intel `Intel(R) Arc(TM) Pro 140T GPU (32GB)`, `8086:7d51:2346:17aa`, revision03, `0000:00:02.0`, driver `32.0.101.x` in `[32.0.101.8517,33.0.0.0)`;
- NVIDIA `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, `10de:2d39:2346:17aa`, revisionA1, `0000:01:00.0`, display driver `[595.71,600.00)`, CUDA device0.

Ambiguity is blocked. Full raw inventory/versions/capabilities are evidence.

Order: CPU and all controls → Intel → full Intel cleanup → NVIDIA → full NVIDIA cleanup → independent verifier. No overlap.

Intel host-USM buffers: record675,840, input4,096, output1,024, uint32 counters2,048 = **683,008 bytes**. Before launch CPU directly initializes output words to `0xffff`, counters to zero, and writes pristine record/input. `clSetKernelArgMemPointerINTEL` only; `cl_mem` and all OpenCL copy/map/migrate calls zero. `clGetMemAllocInfoINTEL` proves host type/base/size.

NVIDIA pinned buffers and device buffers each use those same four sizes = **683,008 bytes each**, combined **1,366,016**. One nondefault stream executes exactly: `cudaMemsetAsync(output,0xff,1024)`, `cudaMemsetAsync(counters,0,2048)`, H2D record, H2D input, one kernel, D2H output, D2H counters, one synchronize. Thus memset2/H2D2/kernel1/D2H2/sync1. No dense/dequantized matrix exists.

## Safe checker and deterministic controls

The only safe dispatch checker has this global order:

`exact byte size → header structural schema → CRC → exhaustive field scan (all <=30) → canonical source/codes/scales/record digests → requested identity → input identity/digest → dispatch`.

There is no expected-digest override. Every control derives from the pristine frozen bytes, is retained, invokes the same checker, and leaves both device submission counters zero.

1. Truncate record to 675,839 bytes: reject `size`.
2. Expert50→51 with otherwise pristine payload and recomputed structural bytes/CRC: pass structure/CRC/field scan/digests appropriate to header-excluded payload, then reject `requested_identity` before dispatch.
3. Code byte index5 `0x32→0x33`, stale CRC: reject `crc`.
4. Same code-byte mutation with CRC recomputed to `2,106,093,510`: pass CRC/fields, reject `canonical_codes_digest`; mutated codes SHA `0f980e086029b664b4d7705349bb184732f35ccd22707623cb8aa49bcd59fdd9`.
5. Scale word0 `0x3b14→0x3b15`, CRC recomputed to `3,896,522,641`: pass CRC/fields, reject `canonical_scales_digest`; mutated scale SHA `df46052d03f183db7029478f0e715b21862378e837d30a6f3b1a572810a907a5`.
6. Natural input word0 `0xbe34→0xbe35`: record passes; reject `input_digest`.
7. Field31: row0/group0/slot0 stored field `23→31`, first pack `df4fe56065`, CRC recomputed `3,350,049,836`; checker rejects specifically `field31` before any canonical digest adjudication. Mutated record SHA `b7cd621738866a88f75ef1c9c70ead443f9b5bbe32cd9ca3147b716bc3558701`.

### Deterministic q sensitivity selector

Selection uses only pristine frozen codes/scales, never natural output:

1. lexicographically enumerate `(row,group,slot)`;
2. take the first decoded q that is nonzero;
3. mutate exactly one step toward zero, preserving `[-15,15]`;
4. keep all other fields/scales/header identity unchanged and recompute CRC.

This deterministically selects `(row0,group0,slot0,column0)`, q `8→7`, stored field `23→22`, first pack `d74fe56065→d64fe56065`. Mutated CRC is `356,698,878`; codes SHA `86f2470ba33598306b388cb2762686e3cd5b140e90eff5289606003fbd83b224`; header SHA `6b5b3dd4017914700c6e4ee9a540a5d88c72dd06b2d400a352e5ce973e75d77c`; record SHA `b5ea50e57b049ba50cdbfc4293272f7d3715af3676e2f2154467f45c79c7e0d8`. The safe checker rejects `canonical_codes_digest` after CRC/field scan.

The sensitivity witness is CPU-only and synthetic, separate from the natural correctness arm. Activation is BF16 `[2048]`, exactly one nonzero at column0, amplitude `2^k`, other words +0. Enumerate integer `k=-8..8`; choose the first whose closed-form `BF16(BF16(decoded_weight)*BF16(2^k))` words differ. This freezes `k=-8`, amplitude word `0x3b80`, activation SHA `2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561`; original/mutated decoded weight words `0x3c94/0x3c82`; expected output row0 words `0x3894/0x3882`. Full 512-word original/mutated SHA are `ca913e50693d83329869fb61dabb75467df7091d39e9a5dd9e17e8480bbeb9f6` and `1868bd78f7059362bed974138ae89c4efa7b930fdf3d07db11db6cd94677ee23`; exactly one word differs. The natural D2 mutation witness, if reported, is diagnostic only and never a gate.

## Positive conjunction and boundary

- All source/input/device/codec/oracle bindings and all controls pass at exact stages with zero control dispatches.
- CPU, Intel, NVIDIA each retain exactly 512 finite uint16/BF16 words and counters `[512]` all one; no `0xffff` output sentinel remains.
- CPU=Intel=NVIDIA bitwise: zero different words, equal SHA.
- Intel forbidden copies zero; NVIDIA exact memset2/H2D2/kernel1/D2H2/sync1.
- Intel fully releases before NVIDIA init; each owned release attempted once and succeeds.
- Start RAM≥2GiB, peak working set≤2GiB, NVIDIA free VRAM≥64MiB, exact memory totals above, evidence≤4MiB, no persistent record/bank.
- Independent verifier reconstructs all bytes, arithmetic, control selection/witness, calls, counters, identities, resources, cleanup and transaction, trusting no aggregate Boolean.

Any false conjunct is negative/blocked/valid-failure; no retries or relaxed equality.
