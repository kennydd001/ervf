# HET-NEXT-CAP0P — process-isolated dual-device cohabitation

## Status and claim

CAP0P is a clean standalone capability experiment. It inherits no CAP0 implementation and contains no model, checkpoint, Q5, routing, quality or performance work. Execution remains closed pending independent source audit.

The sole positive claim is: one coordinator launched one Intel child and one NVIDIA child under a Windows kill-on-close Job Object; both device processes were simultaneously alive and crossed the same start barrier for exactly three repetitions of a 1024-word uint32 sentinel; all six raw outputs were exact and both children exited zero with no surviving child PID.

This is not a resource-by-resource leak proof. Each child attempts explicit cleanup, while process isolation and Job Object termination bound any driver/runtime residue to child lifetime.

## Frozen data and kernels

Seed `0x4845544E45585430`; 1024 little-endian uint32 words use `x[0]=0x45585430`, `x[i]=(1664525*x[i-1]+1013904223) mod 2^32`.

- Intel: `rotl32(x xor 0xA5A5A5A5,7)+0x3C6EF372 mod 2^32`.
- NVIDIA: `rotr32(x+0x9E3779B9 mod 2^32,11) xor 0xC3C3C3C3`.

Expected SHA-256: input `a9d32afd712f6ac80ef7739b11c2baa59e4f84c2067e20307f175de4e8a1acca`; Intel `c83e434be87333bc6bf15d3f0ee492c3e3f9d65b847902bea55310165a42923f`; NVIDIA `f07c3d87d952d1dc82c65d90f467af87426c1658267b7d94f359122e73eafd5f`.

Intel child uses the independently established OpenCL Intel host-USM mechanism: exactly 4096 bytes `clHostMemAllocINTEL`, direct CPU write/read, `clSetKernelArgMemPointerINTEL`, no cl_mem/read/write/migrate. NVIDIA child uses CuPy RawModule/NVRTC, one explicit nondefault stream, 4096-byte pinned host and device buffers, explicit H2D/kernel/D2H. Both retain identities, PCI strings if exposed, compilation log/source/binary hash, raw 1024 output words, per-repetition submit/done monotonic timestamps and cleanup attempts.

## Process protocol

Coordinator is pinned LP0. Child command lines require `--lp 2` Intel and `--lp 4` NVIDIA; each pins and verifies its LP before device enumeration. An optional diagnostics thread is pinned LP6 and records 100ms RAM samples only; it cannot affect adjudication.

Coordinator creates one Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, launches both children hidden with inherited pipes, assigns both process handles to the job, and retains PID/process-create identity. Children emit exactly one create-new ready JSON line after device initialization. Coordinator receives both ready rows, writes one `START 1` line to each pipe, then epochs 2 and 3 only after both preceding result lines. Each child emits one JSON result line per epoch and remains alive across all epochs. Coordinator retains send/receive timestamps. Strict overlap per epoch is `max(intel.submit,nvidia.submit) < min(intel.done,nvidia.done)`; it proves concurrent lifetime/work interval, not speed.

After epoch 3 coordinator sends `STOP`. Children explicitly cleanup and emit one cleanup JSON before exiting zero. Coordinator waits both process handles, records exits, verifies neither PID is alive, closes pipes/process/thread/job handles, and proves job close occurred. Any timeout, malformed row, nonzero exit, surviving PID, missing cleanup or output mismatch is negative. No retries.

## Evidence and transaction

Output directory must be absent. Coordinator retains child stdout/stderr raw bytes, command lines, ready/repetition/cleanup rows, process and Job Object lifecycle, all raw output words and hashes. Success writes create-new result then commit-last with fsync/write-through. Failure first closes the job/waits and then writes create-new failure evidence with PIDs, exits and dispositions. Valid existing commit returns `already_complete` without a failure write; stale output is quarantined and execution aborts.

Positive requires exact 3×2 repetition cardinality, zero word differences, exact hashes, strict overlap all epochs, both exit zero, no surviving PID, job close and independent verifier pass. No throughput/model/quality claim follows.
