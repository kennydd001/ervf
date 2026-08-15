# HET-NEXT-L0-C0-R1 — capability and preflight design

This document is design-only. It authorizes no executable preflight, compiler, device enumeration, allocation, queue, model/checkpoint payload run, kernel launch or timing. C0-R1 implementation requires another independent design GO first.

## Phase 0 — static, no device and no tensor payload

The future standalone Phase-0 preflight must import no CUDA, OpenCL, Level Zero, SYCL, Torch, Safetensors or model runtime. A minimal parser may read JSON/Markdown/source, file metadata, safetensors headers and the separately committed public four-row route-ID metadata only. It must not read D2 or shard tensor payload.

It must verify:

1. exact self, runner, independent verifier, preregistration, schedule generator, guarded range reader, kernel/compiler source and lock hashes;
2. all immutable D2-R3, R5, C1-R2A, official shard, ST2, D7 and C0 audit provenance;
3. shard exact size and prior full-SHA declaration, deferring a fresh 4 GB hash to authorized source-build;
4. exact public route IDs but no p1-p3 tensors, weights, shared-gate, oracle or control arrays;
5. source AST fixes rank ownership while deriving BF16 accumulation in ascending expert ID, with every cast/add point represented and no mutable alternative;
6. validation-seal state machine: p0-only guarded offsets, zero test-range reads before atomic pass, separate clean post-pass test source process and immutable access ledger;
7. exact seed `2026081302`, T0/T1/T2 templates, 30-block/360-observation schedule, 120 samples/arm, 30 warmup observations/10 per arm, A/B `ABBAABBA` projection, position balance, pair IDs and canonical schedule digest;
8. independent implementation of the frozen NumPy-linear quantile formula and the exact four performance formulas;
9. inclusive host-wall boundaries around submissions, sample copies, waits and host merge;
10. exact 256 MiB thrash initialization/first-touch/stride/start-line/mutation/checksum/digest contract;
11. exact English PDH paths, 100 ms monitor cadence, pre/post window and three paging thresholds; exact warmup clock-baseline and live five-sample/70% rule;
12. controls, resource/thermal/device/cleanup gates and atomic transaction schema;
13. independent verifier imports no runner, kernel builder, codec, schedule, quantile, range-reader or transaction helper;
14. every output path is absent;
15. TEMP-only executable simulations of p0 fail sealing tests, p0 pass opening tests, forbidden test-offset rejection before read, exact schedule/statistics on fixed arrays, paging and clock boundary cases, atomic success/failure/recovery, cleanup equality and verifier nonzero exit on one false conjunct.

The static result is hash-bound and cannot authorize Phase 1.

## Phase 1 — separately authorized capability-only process

After independent Phase-0 source GO only, a separate capability process may enumerate and compile minimal no-weight sentinels. It reads no checkpoint/D2 payload and performs no performance timing. It captures:

- exact Intel and NVIDIA PCI/device/driver/runtime/compiler identity;
- Intel host-USM, required subgroup width, profiling, allocation limits and cache inventory;
- NVIDIA integer/BF16 primitives, pinned/device-buffer/event and concurrent work capability, VRAM and cache inventory;
- CPU/NUMA/topology identity, the timing thread's node and both device topology relations;
- simultaneous distinct-device usability;
- compile logs and source/binary hashes;
- at most `1 MiB` allocation per device, followed by exact cleanup and unchanged handle/memory counts;
- proof the frozen 256 MiB thrash buffer exceeds every reported relevant last-level cache.

Any missing capability is `blocked_capability`. Phase 1 cannot authorize Phase 2 or execution by itself.

## Phase 2A — separately authorized p0-only CPU source build

Only after capability PASS may a CPU-only clean process:

1. rehash the official shard and D2 file;
2. use the guarded range reader to open only p0-authorized D2 tensors;
3. read/quantize only p0-required official expert triplets plus shared;
4. independently requantize/redecode and hash every p0 code/scale/source binding;
5. build and retain only p0 oracle/control manifests and compact arrays;
6. prove from the access ledger that no p1-p3 payload or test-only shard range was read.

No device is opened. Records are anonymous/temporary and no persistent bank is allowed.

## Validation source audit and physical lock

Before one physical p0 attempt, a new independent audit must verify real-weight tensor arithmetic, official expert-ID-sorted BF16 accumulation, Intel/NVIDIA kernels, fixed ownership, safe controls, timing schedule/bounds, thrash/PDH/clock rules, resources, transactions and cleanup. Only a new immutable execution lock may authorize the p0 attempt.

## Phase 2B — test source build after committed p0 PASS only

If and only if independent verification commits the p0 pass, a separate clean CPU process may consume that exact commit, change `tests_opened` atomically, and read p1-p3 payloads and additional official weights. It must produce per-row source/oracle/access manifests before device timing. It may not change source, kernels, split, schedule, thresholds or resource gates. Tests execute once under the already frozen state machine; no new performance-derived lock or retuning is permitted.

At every phase, a blocked/negative result creates immutable evidence and closes later phases. No capability or device action is authorized by the present documents.
