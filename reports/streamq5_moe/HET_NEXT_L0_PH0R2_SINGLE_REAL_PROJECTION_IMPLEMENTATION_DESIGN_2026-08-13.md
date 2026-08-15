# HET-NEXT-L0-PH0-R2 — implementation design

Date: 2026-08-13  
Status: **source design only; no runner, preflight, import, compiler or device call authorized**

Normative protocol: `HET_NEXT_L0_PH0R2_SINGLE_REAL_PROJECTION_PREREGISTRATION_2026-08-13.md`. R1 is immutable and superseded only for checker order, control selection/witness and sentinel details.

## Future standalone modules

Only after design GO may a new revision create: builder; helper-independent software-FP32 CPU oracle; OpenCL host-USM backend; CUDA tiled-width8 backend; sequential phase runner; independent verifier importing none of them; locks; and static no-payload/no-device preflight.

Create-new states are `EMPTY→CPU_COMMITTED→INTEL_COMMITTED→INTEL_CLEAN→NVIDIA_COMMITTED→NVIDIA_CLEAN→VERIFIED`. Failure cleans before durable evidence; valid completion returns `already_complete`; no automatic physical retry.

## Fixed implementation facts

- Read only 2,097,152 matrix bytes and 4,096 natural input bytes.
- Build one anonymous 675,840-byte record and exact independent 512-word CPU oracle.
- Intel host-USM sizes `675840/4096/1024/2048`, initialize output `0xffff` and counters zero through host pointers; no explicit copies.
- NVIDIA pinned/device sizes identical; initialize device output/counters via two `cudaMemsetAsync` calls; ledger H2D2/kernel1/D2H2/sync1.
- Launch 16×256, 32 width8 rows/block, row `block*32+thread/8`; OpenCL subgroup8; CUDA `cooperative_groups::tiled_partition<8>` shuffles 4/2/1.
- Unpack every field; never detect field31 via constant five-byte sentinels.

## One checker, one selector

Safe checker source has a single entry point and frozen order: size, structural header, CRC, exhaustive field scan, canonical digests, requested identity, input digest, dispatch. It has no optional/override expected hashes.

The q sensitivity selector is a pure function of pristine codes/scales: lexicographic first nonzero q, one step toward zero. Source/unit tests assert the fixed result row0/group0/slot0 q8→7 and all fixed mutated hashes. The one-hot witness independently reconstructs decoded weights, enumerates k from -8 to 8, selects -8, and proves the fixed activation/output words and full SHAs. Device backend symbols cannot call unsafe witness functions.

## Independent verifier

After device cleanup it rereads only the two permitted ranges; independently requantizes, packs, parses, scans, and computes software-FP32 oracle; reconstructs every control including its adjudication stage and the selector/witness closed form; rehashes exact raw arrays; compares 512 words/counters; and validates exact devices, allocations, calls, sequential lifecycle, releases, resources and atomic artifacts. It imports no runner/builder/backend/codec helper and exits nonzero on any false conjunct.

## Static preflight obligations

The future preflight imports no device/compiler/model module and reads no payload. It binds all docs/source/locks; derives `675840`, `683008`, `1366016`, 16×256 and call counts; AST-checks no model/router/shared/merge/performance/bank path, exact checker order/no override, tile8/subgroup8, field scan and forbidden Intel copy APIs; unit-tests every q/tie/zero/packing/CRC/control/selector/witness/FMA/add case; simulates exact counters and sentinels; fault-injects all cleanup and transaction positions; mutates all verifier gates; and proves execution closed/output absent.

Design/preflight success does not authorize device execution.
