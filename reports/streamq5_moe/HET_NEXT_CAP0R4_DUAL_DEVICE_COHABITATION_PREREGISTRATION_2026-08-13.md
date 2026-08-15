# HET-NEXT-CAP0-R4 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â dual-device cohabitation capability gate

This immutable R2 supersedes but does not alter CAP0's scientific scope. NVIDIA ownership is exclusively direct CUDA Driver API plus direct NVRTC; no CuPy/runtime/default-stream/primary-context ownership is permitted. It also adds executable stale-ack/timeout/release/transaction negatives, ABI and PCI identity gates, a 2 GiB start-RAM gate, cache-line-separated epoch cells, real release calls, raw word retention, independent reconstruction and global cleanup evidence. CAP0 and R1 remain immutable and unexecuted.

## Status and sole claim

This is a brand-new standalone preregistration. It inherits no C0/C0-R1..R6 scientific or execution protocol. It contains no D2, checkpoint, shard, model, Q5, routing, numerical-quality, performance or test-seal input. It authorizes no execution until source audit and a new authorization-only lock revision.

The only admissible positive claim is:

> In one process with four fixed persistent host threads, one Intel GPU host-USM queue and one NVIDIA CUDA stream coexisted and concurrently executed three exact 4 KiB integer sentinels with zero word differences, valid lifecycle evidence and exact cleanup.

No latency, bandwidth, throughput, model, quality, heterogeneous-compute benefit or breakthrough claim is allowed.

## Frozen host topology and lifetime

Capability must verify Windows logical processors 0, 2, 4 and 6 exist, are in the same processor group and map to four distinct physical cores. Otherwise `blocked_topology`; no fallback affinity.

- LP0: persistent coordinator; sole result assembler.
- LP2: persistent Intel worker; sole owner of Intel context and one in-order profiling queue.
- LP4: persistent NVIDIA worker; sole owner of one CUDA device/context/stream.
- LP6: persistent PDH monitor; never calls a device API.

All four threads start before device enumeration and remain alive through three repetitions and cleanup. Their process/thread IDs, process start identity, processor group/logical processor/core identity, start/end QPC and roles are retained. Intel and NVIDIA identities must have distinct PCI bus/device/function values.

## Exact input and bijections

`SEED = 0x4845544E45585430`. The 1024 little-endian uint32 input words are independently reconstructible:

```
x[0] = low32(SEED) = 0x45585430
x[i] = (1664525 * x[i-1] + 1013904223) mod 2^32, i=1..1023
```

Intel output word `yI[i]` is:

```
rotl32(x[i] xor 0xA5A5A5A5, 7) + 0x3C6EF372 mod 2^32
```

NVIDIA output word `yN[i]` is:

```
rotr32(x[i] + 0x9E3779B9 mod 2^32, 11) xor 0xC3C3C3C3
```

Both maps are bijective over uint32. Independent verifier regenerates all input/expected words without runner helpers. Each repetition requires exact 1024/1024 equality, zero different words, and exact input/output little-endian SHA-256. Outputs from all three repetitions must be byte-identical per device.

## Intel arm

Enumerate all OpenCL GPU devices and require exactly one Intel Arc device satisfying `cl_intel_unified_shared_memory`. Retain platform/device name/vendor/version, driver, OpenCL C version, extensions, PCI information when exposed, global memory, max allocation, address bits, queue properties and all host-USM capability bitfields.

Create context and one in-order profiling queue. Allocate exactly 4096 bytes with `clHostMemAllocINTEL`, alignment 4096. Verify allocation type is host, base matches and allocation size is 4096 using `clGetMemAllocInfoINTEL`. CPU writes input words directly to the returned pointer. Bind only with `clSetKernelArgMemPointerINTEL`. The code path must contain no `cl_mem`, `clCreateBuffer`, `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`, memcpy-to-device or migrate/prefetch call. Kernel transforms words in place. CPU reads the same pointer after `clFinish`.

Compile flags are exactly `-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt`. Retain full build log even on success, source SHA and retrievable program binary bytes/SHA. Every repetition retains profiling start/end ns.

## NVIDIA arm

Require exactly the selected NVIDIA CUDA device and retain name, UUID, PCI identity, driver/runtime/NVRTC versions, compute capability, total memory, concurrent-kernel/copy and host-memory capability bits.

Allocate exactly 4096 bytes CUDA pinned host memory and exactly 4096 bytes CUDA device memory. CPU writes input words into pinned memory. In the persistent CUDA stream enqueue H2D 4096 bytes, the fixed uint32 kernel, and D2H 4096 bytes. Synchronize the stream. No default stream or managed memory is permitted.

NVRTC flags are exactly `--std=c++14 --fmad=false`. Retain full compile log, source SHA and compiled binary bytes/SHA. Each repetition retains CUDA event start/end telemetry, diagnostic only.

## Exact concurrent protocol

Only Win32 `CreateEventW`, `WaitForSingleObject`, `WaitForMultipleObjects`, `SetEvent`, `ResetEvent`, `SRWLOCK`, `MemoryBarrier`, `InterlockedExchange64`, `InterlockedCompareExchange64` and QPC implement the protocol. Python condition/Event/futures/thread pools are forbidden for device coordination.

Per worker: auto-reset `command_i`; manual-reset `ready_i`, `done_i`, `stop_i`; cache-line-separated uint64 `last_command_epoch_i`, `ack_epoch_i`. Common `start` is manual-reset. Before each repetition, coordinator confirms exact prior ack, resets common/active events, publishes immutable descriptor under exclusive SRWLOCK, publishes epoch, signals Intel command then NVIDIA command, waits both ready, samples QPC `t0`, and performs one `SetEvent(start)`. Both workers run their device work concurrently. After output sync and telemetry write, each worker executes `MemoryBarrier`, publishes ack and immediately signals done. Coordinator waits fixed `[intel_done,nvidia_done]` with `bWaitAll=TRUE`, verifies both acks, reads outputs, samples `t1`, then resets start/ready/done. Timeout is 30 seconds and fail-closed; no retry.

Run exactly three repetitions at epochs 1, 2 and 3. Both queues/streams and all four threads remain alive across all repetitions. Retain every primitive call/return, event identity/state, epoch, QPC and queue/stream identity. The intervals `[intel_submit,intel_done]` and `[nvidia_submit,nvidia_done]` must strictly overlap in each repetition; this proves concurrent lifetime/work overlap, not performance benefit.

## PDH diagnostic monitor

LP6 owns one PDH query with English counters `\Memory\Page Reads/sec`, `\Memory\Pages Input/sec`, `\Paging File(_Total)\% Usage`. It begins at least 500 ms before repetition 1 and ends at least 500 ms after cleanup. A waitable timer schedules 100 ms samples; retain scheduled/actual QPC, status and three values. Exact validity: at least 11 samples, every interval in `[80,120] ms`, lateness `<=20 ms`, all counter statuses valid. PDH is diagnostic only; values cannot make a correct capability result positive or negative. Invalid cadence/status is `invalid_monitor_protocol`.

## Lifecycle, resources and evidence

Start available RAM must be at least 2 GiB. NVIDIA free VRAM must be at least 64 MiB. Total sentinel allocation is exactly Intel 4096 + NVIDIA pinned 4096 + NVIDIA device 4096 bytes. No weight/model file path may exist in source constants or be opened.

Every create/allocate/register/compile/queue/stream/event/PDH handle has one ledger row with unique ID, creator thread, create QPC, release-attempt QPC, return code and final state. All releases are attempted in `finally`, even after an earlier release error. Positive result requires each create exactly once, each release attempted exactly once, all release codes success, no nonzero allocation/handle counts, CUDA memory-pool used bytes zero and Intel USM freed.

Output directory must be absent. Success writes create-new temp JSON, fsync, rename; then create-new commit JSON last. Failure first performs cleanup, then writes create-new atomic failure JSON containing stage/error/traceback, all partial ledgers and cleanup dispositions. Stale/uncommitted output is quarantined under `failed_attempts`; valid committed output is never overwritten. No physical retry or retuning.

## Adjudication

- `dual_device_cohabitation_positive`: topology, identities, three concurrent exact repetitions, monitor protocol, provenance and cleanup all pass.
- `capability_negative`: device/extension/ABI/compile/sentinel/concurrency or cleanup condition fails.
- `blocked_topology`, `blocked_resource`, `blocked_device`, `invalid_monitor_protocol`, `invalid_protocol`: named fail-closed outcomes.

Positive CAP0-R4 opens only a separately preregistered real-weight component phase. It says nothing about whether heterogeneous execution is faster.






