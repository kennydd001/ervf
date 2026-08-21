# S100 Phase52 — cold-rotating mapped-host Ornith experts

## Reason for the recheck

Phase51 repeatedly read one 1.6875 MiB pinned expert record. That record fits in
the RTX GPU's 32 MiB L2, so the result is a valid warm-UVA measurement but not a
cache-miss transport decision.

## Frozen correction

- Reuse the Phase51 implementation and checkpoint tensors.
- Rotate every invocation over 80 byte-identical pinned expert records.
- The resulting 135 MiB host working set is required to be at least 4x measured
  GPU L2 capacity.
- Run M1, M2, M4 and M8 against hot-device, rotating staged and rotating direct
  paths. Timings remain one-stream CUDA event intervals.

## Gates

1. Rotating working set is >= 4x measured L2.
2. Every direct output is finite, deterministic and agrees with the resident
   output at normalized RMSE <= 0.001 and normalized max error <= 0.005.
3. Cold direct M8 is <= 0.75 ms.
4. Cold direct M8 is no slower than cold staged M8.
5. Cold direct M1 is no slower than cold staged M1.

## Claim boundary

Passing selects direct mapped-host reads for an uncached single expert record.
It still excludes disk, route-union concurrency, cache policy and whole-model
throughput.

