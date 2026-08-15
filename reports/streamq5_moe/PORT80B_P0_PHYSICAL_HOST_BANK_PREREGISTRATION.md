# PORT80B_P0 - physical full-size host-bank gate (preregistration)

This protocol is frozen before creation of the full-size bank and before any
CUDA allocation or transfer.  It opens only `PORT80B_ACTIVE_SET/P0`; it does
not modify or reopen any earlier registry.

## Question

Can the current 64-GB Windows host keep and stream an exact-final-size,
non-sparse Q5 expert bank without post-warm-up hard paging, excessive process
commit, unstable transfer tails, or thermal failure?

This is a host-residency and transfer gate.  It contains no model weights,
quality measurement, router trace, expert compute, dense shell, attention,
DeltaNet, prefill, or end-to-end decode.

## Frozen physical contract

The dimensions and byte counts are inherited from the independently verified
N4A and N4B-R artifacts for revision
`a19358a7659bd1f564300250ee189120c49a562f` of
`Qwen/Qwen3-Coder-Next`:

- 48 layers;
- 512 routed experts plus one shared expert per layer;
- projection order `gate`, `up`, `down`;
- matrix shapes `(512, 2048)`, `(512, 2048)`, `(2048, 512)`;
- group size 128, 5-bit codes, raw BF16 scales;
- 64-byte `SQ5M` header and 4,096-byte record alignment;
- 675,840 bytes per matrix, 2,027,520 bytes per expert;
- exactly **49,925,652,480 bytes** for the monolithic bank
  (`46.49688720703125 GiB`).

The bank order is layer-major, then expert-major (`0..511` routed, `512`
shared), then projection-major (`gate=0`, `up=1`, `down=2`).  Every header
records its coordinates, shape and payload CRC32 using
`<4sHHHBBIIH2xIII28s`.  Synthetic code bytes are `0x55`, scale words are
BF16 `0x3c00`, and alignment padding is zero.  The file is written
sequentially with real payload bytes and `fsync`; preallocation/truncation,
sparse ranges, NTFS compression and deduplication shortcuts are forbidden.

Before benchmarking, the runner must verify:

1. exact logical file size and full-file SHA256 from the build manifest;
2. no `FILE_ATTRIBUTE_SPARSE_FILE` or `FILE_ATTRIBUTE_COMPRESSED` flag;
3. `GetCompressedFileSizeW >= 49,925,652,480` bytes;
4. deterministic sampled headers, payload CRCs and zero padding;
5. read-only memory-map mode.

Any failure closes P0.  A failed or interrupted build is retained with the
`.inprogress` suffix and is never accepted as a bank.

## Frozen trace and cache protocol

The route generator is counter-based SplitMix64 with seed `0x80B0120826`.
For each `(token, layer)`, it generates one start and one odd stride modulo
512, yielding ten distinct routed experts.  The same route stream and digest
must be used for all scenarios.  It is a deterministic stress trace, not a
claim about the real Qwen router.

After a full sequential SHA256 sweep warms every mapped bank page, run in one
process:

1. `zero_cache`: 10,000 tokens, all 480 routed expert records transferred;
2. `cache_4k`: 10,000 tokens, a physical GPU LRU with 2,420 expert slots;
3. `cache_32k`: 10,000 tokens, a physical GPU LRU with 2,072 expert slots.

The caches are global over `(layer, expert)` and transfer a full aligned
2,027,520-byte record only on a miss.  The zero-cache destination is an
eight-slot rotating device ring.  Host staging always uses exactly eight
CuPy page-locked windows of one expert record each (16,220,160 bytes total).
CUDA events measure transfer-only token time; wall-clock time separately
measures mmap staging plus H2D.  Cache hits, misses and transferred bytes are
reported independently.

If the three 10,000-token scenarios complete in less than one hour, the runner
continues the zero-cache trace in 250-token stability chunks until measured
benchmark wall time reaches 3,600 seconds.  This extension is part of the
same uninterrupted process.

Full `cudaHostRegister` of the 46.5-GiB mapping is optional, is not a pass
condition, and requires a separate explicit command-line flag.  If requested,
its result is recorded and the mapping is immediately unregistered; failure
does not replace the required eight-window path.

## Windows memory and fault telemetry

At load, after warm-up, every 250 tokens, at scenario boundaries, and at exit,
record process RSS/peak working set, private bytes, pagefile/peak pagefile
(Windows process commit), total page faults, system available RAM and swap.

Post-warm-up hard paging is conservatively operationalized with the English
PDH counters `\\Memory\\Page Reads/sec` and `\\Memory\\Pages Input/sec`,
sampled once per second from immediately after the full-bank warm-up through
the end of the run.  These are system-wide physical page-in counters: any
background-process page read therefore makes the gate fail conservatively.
`Process.num_page_faults` is also logged but is explicitly labelled as
hard-plus-soft and cannot by itself prove the hard-fault gate.

GPU temperature, power and SM clock are sampled with `nvidia-smi` at least
every 30 seconds and at each scenario boundary.

## Frozen pass/fail gates

P0 passes only if all conditions hold:

- the physical-bank contract and full SHA256 verification pass;
- all three scenarios complete exactly 10,000 primary tokens;
- total uninterrupted benchmark duration is at least 3,600 seconds;
- every post-warm-up `Page Reads/sec` sample is exactly zero;
- peak Windows process commit (`peak_pagefile`) is at most 58 GiB;
- zero-cache H2D p95 is at most 45 ms per token;
- CUDA reports no allocation, transfer or synchronization error;
- telemetry contains no missing interval longer than 45 seconds;
- no thermal shutdown or driver reset occurs.

Cache-scenario latency and hit rate are descriptive and cannot rescue a
zero-cache failure.  A failure caused by memory pressure, post-warm-up page
reads, pinning failure or H2D p95 above 45 ms authorizes consideration of
96 GB RAM.  It does not authorize a CPU, NPU, GPU or SSD purchase.

## Safety interlock and claim boundary

The runner defaults to `--phase preflight`, which cannot create the bank,
import CuPy, allocate pinned/device memory, or execute GPU work.  Build,
benchmark and full modes require the exact acknowledgement token
`PORT80B_P0_49925652480`.

No physical file or GPU work may be started until the preflight report has
shown the exact disk/RAM impact and a separate timing-go has been given.

A P0 pass proves only that the synthetic final-size expert bank can be hosted
and physically streamed under this protocol on this machine.  It is not an
80B runtime, quality, real-routing, dense-shell, prefill or tokens/s result.
