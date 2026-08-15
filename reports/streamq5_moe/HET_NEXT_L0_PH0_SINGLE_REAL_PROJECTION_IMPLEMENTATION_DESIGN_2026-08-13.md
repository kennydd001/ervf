# HET-NEXT-L0-PH0 single real projection — implementation design

Date: 2026-08-13  
Status: **source design only; no runner, preflight, model load, device API, compile, or kernel call authorized**

This document implements no code. It freezes the smallest auditable path for the preregistered component.

## Planned modules

The next revision may create, on new paths only:

1. a standalone CPU builder/oracle module;
2. an Intel OpenCL host-USM backend;
3. an NVIDIA CUDA backend;
4. one phase runner with explicit `build`, `intel`, `nvidia`, and `adjudicate` states;
5. a truly independent CPU verifier that imports none of the first four modules;
6. locks and a static preflight that imports no OpenCL/CUDA/compiler library and reads no payload.

The implementation source must be independently audited before the static preflight. The preflight must pass before source/input payload access; a separate explicit authorization is required before either device arm.

## State machine

`EMPTY → CPU_COMMITTED → INTEL_COMMITTED → INTEL_CLEAN → NVIDIA_COMMITTED → NVIDIA_CLEAN → VERIFIED`.

- Each transition is create-new, fsyncs files and the containing directory, and records hashes of all prior state.
- A phase refuses unexpected prior or future files. A valid completed state returns `already_complete` without writing a failure.
- Failure evidence is written only after owned resources are synchronized and every cleanup is attempted. Partial artifacts are quarantined and listed; there is no automatic physical retry.
- Device initialization is illegal before `CPU_COMMITTED`. NVIDIA initialization is illegal before `INTEL_CLEAN`.
- The sole in-memory record is reconstituted from the committed CPU manifest and the frozen source range; it is never promoted as a persistent bank.

## CPU phase

The reader first validates the full small manifests and safetensors headers, then reads exactly two payload ranges: 2,097,152 source bytes and 4,096 input bytes. It must not instantiate a Transformers model or map/read any other D2 tensor.

Two independent implementations are required inside the future package:

- builder: exact STREAMQ5 packer producing the 675,840-byte record and all known S0 digests;
- oracle: independent record parser plus software IEEE-binary32 width-8 reduction producing 512 uint16 BF16 words.

They share only immutable byte buffers and constants, not quantize/decode/reduction helpers. The CPU commitment retains source/input/header/codes/scales/padding/record/oracle digests, CRC, q min/max, zero-group count, exhaustive field count, 512 output words, row-write bitmap, runtime/resource samples, and all three negative-control verdicts.

## Kernel mapping

Both device sources literally encode the same map:

```text
local size = 256
width = 8
logical rows per group/block = 256 / 8 = 32
groups/blocks = ceil(512 / 32) = 16
subgroup/warp-group = local_thread / 8
lane = local_thread % 8
row = group_or_block * 32 + subgroup
pack(lane,v) = lane + 8*v, v=0..31
```

The only legal global launch is 16×256. A 512-byte row-write counter/bitmap initialized to sentinel values accompanies validation (not the matrix output); the kernel atomically increments its own row counter exactly once. This diagnostic must not enter arithmetic.

The source freezes explicit helper functions for BF16-to-FP32, round-to-nearest-even BF16, little-order 5-bit unpack, FP32 FMA, and the exact add tree. OpenCL and CUDA compiler flags must disable fast math and retain the explicit order. Kernel source and compiler binary/log bytes are hashed before accepting output.

## Intel backend

The implementation is an audited 512×2048 specialization of `run_st2_mini_ergv_w8.py`, not an import or textual parameter substitution at runtime. Every OpenCL and Intel-USM function receives exact Windows ABI `argtypes/restype`, and every return is checked.

Allocation ledger:

| Buffer | Type | Bytes | Access |
|---|---:|---:|---|
| record | Intel host-USM | 675,840 | CPU write, kernel read |
| input BF16 | Intel host-USM | 4,096 | CPU write, kernel read |
| output BF16 | Intel host-USM | 1,024 | kernel write, CPU read after finish |
| row counters | Intel host-USM | 2,048 | CPU zero, kernel atomic write, CPU read |

The preregistered three semantic buffers total 680,960 bytes; counters are separate diagnostic evidence. There is no `cl_mem` object and no OpenCL copy, map, unmap, or migrate operation. Pointer type/base/size are checked with `clGetMemAllocInfoINTEL`. Cleanup continues after individual errors in order: kernel event, diagnostic/semantic USM allocations, kernel, program, queue, context. All attempts and codes are retained.

## NVIDIA backend

The CUDA specialization uses the same 16×256 launch and width-8 logical groups. It consumes the wire record directly. The input stays BF16 until per-element widening inside the kernel, and output is uint16 BF16.

| Buffer | Pinned host bytes | Device bytes | Transfer |
|---|---:|---:|---|
| record | 675,840 | 675,840 | one H2D |
| input BF16 | 4,096 | 4,096 | one H2D |
| output BF16 | 1,024 | 1,024 | one D2H |
| row counters | 2,048 | 2,048 | diagnostic D2H only |

The semantic pinned/device totals are each 680,960 bytes. The diagnostic counter copy is separately labelled and cannot be counted as a weight/input/output transfer. Allocation and transfer offsets/sizes are exact; no decoded/dense matrix exists. The nondefault stream synchronizes before output inspection. CUDA events, if used for completion, are lifecycle-only and no elapsed time is reported. Cleanup attempts every event, allocation, module, stream, and owned context/runtime object and records the actual return codes.

## Independent verifier

The verifier must:

1. rehash all frozen docs, sources, evidence, shard, D2 header, and output artifacts;
2. inspect safetensors headers and independently prove the exact source/input ranges;
3. reread only those ranges;
4. independently requantize expert 50 gate, reconstruct every wire byte, and reproduce the four known S0 digests plus committed record/CRC/header/padding values;
5. independently parse all 1,048,576 fields and reject 31;
6. rebuild the width-8 software oracle without importing runner/device/codec helpers;
7. require exact raw schema and recompute every raw tensor/byte manifest;
8. compare all three 512-word outputs bitwise, independently count mismatches and row writes, and recompute their SHA-256;
9. reconstruct each control mutation from the pristine committed record and verify rejection order plus zero submissions;
10. reconstruct Intel USM and NVIDIA allocation/copy/launch ledgers, validate non-vacuous exact cardinalities, device identities, sequential lifecycle, resource limits, cleanup, failure dispositions, and atomic commit;
11. exit nonzero for any false conjunct and never accept a runner-provided aggregate Boolean as evidence.

The verifier may report only `positive_single_real_projection_component`, `negative_single_real_projection_component`, `blocked_capability_or_resource`, or `valid_failure_evidence`. Only the first status satisfies the preregistered conjunction, and even that remains validation-only.

## Static preflight obligations

Before any physical authorization, a source-only preflight must:

- AST-check that no model/Transformers forward, router, shared, merge, timing benchmark, or persistent-bank path exists;
- bind preregistration, design, runner, verifier, both device sources, compiler sources, dependency versions, and lock content by SHA-256;
- validate all record/shape/byte/grid constants and independently derive `675840`, `680960`, and `16×256`;
- unit-test 5-bit pack/unpack including every q `[-15,15]`, ties-even cases, zero groups, little order, CRC, padding, and field-31 rejection;
- unit-test the exact reduction DAG with signed zero, subnormal, cancellation, infinity/NaN rejection, and FMA-sensitive fixtures;
- parse ASTs to cover every OpenCL/CUDA API signature, checked call, forbidden Intel copy symbol, allowed CUDA transfer count, and cleanup path;
- execute CPU-only fake-pointer lifecycle, negative-control, atomic-transaction, failure/quarantine, verifier-false-conjunct, and row/grid mapping simulations;
- prove output directories absent and locks explicitly execution-closed.

It must not import OpenCL, CUDA, CuPy, NVRTC, a model library, or read the 2 MiB/4 KiB payload ranges. Passing it authorizes nothing beyond a later independent source audit.
