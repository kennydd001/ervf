# PORT80B-D3 — mapped-host one-kernel preregistration

**Frozen before physical execution:** 2026-08-12

D2 proved that mapped/read-only host registration and a small physical GPU
read work, but 480 copy-engine submissions from a 60–80% registered bank reach
only about 18.6–18.7 GB/s and miss the frozen 45-ms transport gate. D3 asks if
one coalesced GPU kernel can issue the same 480 remote reads with less dispatch
and copy-engine scatter overhead.

## Frozen protocol

- same immutable P0 bank and manifest SHA contract;
- register the first 307 routed experts of every layer (27.826 GiB, 60%);
- no CPU bounce buffer and no dynamic per-token registration;
- a device-resident table contains the 480 mapped device pointers for every
  timed route;
- the kernel performs one remote byte load and one HBM byte store for every one
  of the exact 973,209,600 active bytes;
- all destination bytes are structurally checked after a physical kernel run;
- validation schedules: 512, 1024, 2048 and 4096 blocks, all 256 threads;
- 6 warm-ups and 24 event-timed validation samples/schedule in rotating order;
- choose the lowest validation p50, ties by fewer blocks;
- if correctness holds and selected validation p50 <=65 ms, run once-only test
  on 120 disjoint route tokens; no retune after test.

## Gates

Primary P2 mechanism pass:

- zero destination-byte mismatches;
- 120 finite test samples;
- test p95 <=65.0 ms;
- effective remote payload bandwidth at p95 >=15.0 GB/s;
- 48 registration ranges, clean unregister and no CUDA/runner error.

Strong transport gate: test p95 <=45.0 ms and >=21.627 GB/s. Page-read
telemetry is reported independently; a system-wide PDH event is not silently
attributed to this process.

The run is a 60%-bank mechanism test because D2 already proved that full-bank
registration fails on the current 64-GiB host. It cannot become a full-bank
pass.

## Claim boundary

This copies synthetic Q5-record bytes but performs no Q5 arithmetic. It is not
an exact ERGV host-execution kernel, real 80B inference, quality result,
end-to-end tokens/s or endurance result. A pass only authorizes the exact Q5
compute integration.
