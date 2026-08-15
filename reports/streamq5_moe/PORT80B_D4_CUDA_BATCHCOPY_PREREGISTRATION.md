# PORT80B-D4 — native CUDA batch-copy preregistration

**Frozen before physical execution:** 2026-08-12

The installed CUDA 13.2 runtime exports `cudaMemcpyBatchAsync`, and its bundled
official header defines the exact ABI. D4 tests that API directly; it does not
replace it by a loop or an inferred emulation.

## Frozen protocol

- immutable P0 physical bank and manifest contract;
- same stable 60% per-layer registered/read-only prefix as D2/D3R: 307 routed
  experts/layer, 27.826 GiB;
- 480 records and exactly 973,209,600 bytes per route token;
- three arms on identical routes and one non-legacy CUDA stream:
  - `ordinary480`: 480 `cudaMemcpyAsync` submissions;
  - `batch48x10`: 48 native batch calls with ten independent records each;
  - `batch1x480`: one native batch call with all 480 records;
- native `cudaMemcpyAttributes.srcAccessOrder = Any`; fixed host/device pointers,
  default location hints and flags;
- 6 warm-ups/arm, 24 validation samples/arm in rotating/reversed order;
- candidates are only the two native batch arms; select lowest validation p50,
  ties by fewer copies per batch (`batch48x10` first);
- a candidate opens 120 once-only test samples when full-byte correctness is
  exact and its validation p50 is no worse than 1.05x the same-run ordinary
  control; no retuning after test.

All 973,209,600 output bytes are structurally validated for every arm before
timing. All descriptor arrays remain alive until stream completion.

## Gates

Primary pass:

- native symbol found and ABI structure sizes match the bundled header;
- zero full-buffer byte mismatches for every arm;
- 120 finite selected-candidate test samples;
- selected test p95 <=45.0 ms and >=21.627 GB/s;
- selected p50 and p95 ratios versus same-run ordinary control <=0.90;
- 48 registered ranges, clean unregister, no native CUDA/CuPy/runner error.

If native calls return an error, it is a compatibility result and timing stays
closed. This is still only a 60%-bank transport test because full registration
already failed in D2.

## Claim boundary

No Q5 arithmetic, real 80B checkpoint, model quality, dense shell, tokens/s,
full-bank capacity or endurance is measured. CUDA batch copy is prior art.
