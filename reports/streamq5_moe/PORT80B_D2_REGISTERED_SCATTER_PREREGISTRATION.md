# PORT80B-D2 — registered-bank scatter preregistration

**Frozen before physical execution:** 2026-08-12

## Question

Can the existing 46.496887-GiB physical PORT80B bank be registered in stable
per-layer ranges so that 480 selected expert records move directly from the
read-only file mapping to one device buffer, without `mmap -> pinned` staging?

The parent P0 result remains negative. D1 separately established a 39.041-ms
contiguous H2D p95 and an 88.475-ms page-resident gather p95; D2 tests a new
source-memory path and cannot rewrite either parent result.

## Locked data and sweep

- bank: the existing non-sparse 49,925,652,480-byte P0 bank;
- expected bank SHA-256 from its verified manifest:
  `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`;
- 48 layers, 512 routed experts/layer, ten records/layer/token;
- record size: 2,027,520 bytes; active bytes/token: 973,209,600;
- registered per-layer expert prefixes: 60%, 70%, 80%, 100%, concretely
  307, 358, 410 and 512 routed experts/layer;
- registration flags: mapped + read-only; no writable mapping and no pinned
  bounce buffer;
- deterministic unique top-10 routes inside each frozen prefix;
- 10 warm-ups and 120 physical event-timed samples per successfully registered
  size, with a different route token per sample;
- all 973,209,600 destination bytes are checked structurally after a physical
  transfer: every header byte, Q5 code byte, BF16 scale byte and padding byte.

Registration proceeds from small to large. Every range is unregistered in a
`finally` block. A CUDA allocation/registration failure is a measured capacity
outcome, not a reason to change the sweep. The process stops expanding only on
failure, CUDA/driver error, or less than 2.0 GiB reported available physical RAM
immediately after registration. No retuning after timing.

## Capability probe

Record before the sweep:

- compute capability and discrete/integrated status;
- `canMapHostMemory`, `hostRegisterSupported`,
  `hostRegisterReadOnlySupported`, `canUseHostPointerForRegisteredMem`,
  `pageableMemoryAccessUsesHostPageTables`, unified addressing and async-engine
  count;
- a 64-MiB mapped/read-only registration and `pointerGetAttributes` probe;
- a physical mapped-host kernel read compared byte-for-byte with the source.

TMA and `cudaMemcpyBatchAsync` are not silently approximated. If the local
runtime does not expose a tested binding, they remain separate conditional
phases.

## Frozen gates

For each successfully registered prefix:

1. 48/48 registration ranges succeeded and were later unregistered;
2. full destination structural mismatch count is zero;
3. 120 finite event samples exist;
4. direct-scatter H2D p95 <= 45.0 ms;
5. effective payload bandwidth at p95 >= 21.627 GB/s;
6. no CUDA/runner error or driver reset;
7. post-warm timing telemetry has zero observed system Page Reads/sec when the
   local PDH sampler is available.

A **mechanism pass** requires gates 1-6 at any prefix. A **full-bank pass**
requires all seven gates at 100%. A smaller-prefix pass is evidence for the
mechanism and registration capacity only; it is not a full 80B transfer result.

## Claim boundary

This is a synthetic byte-transport experiment. It contains no real 80B
weights, Q5 arithmetic, dense shell, routing quality, end-to-end tokens/s or
one-hour endurance. Host registration and mapped-host access are CUDA prior
art. A pass authorizes an exact direct-host ERGV kernel and a longer endurance
test; it is not an industrial-breakthrough claim.
