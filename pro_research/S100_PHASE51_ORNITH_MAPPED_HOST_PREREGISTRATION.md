# S100 Phase51 — direct mapped-host Ornith experts

## Question

For an Ornith expert cache miss, is it faster to execute directly against its
1.6875 MiB mapped pinned host record than to copy the full record into a device
staging buffer before execution?

## Frozen input

- Pottokao layer-20 routed expert 0 and deterministic Phase51 hidden rows.
- Complete gate/up/SiLU/down expert at multiplicities M1, M2, M4 and M8.
- `hot`: resident device record, no transfer.
- `stage`: six async H2D copies of codes/scales followed by the resident kernel.
- `direct`: the same kernel reads checkpoint codes/scales through UVA pointers
  into CuPy mapped pinned memory; no H2D copy or device weight mirror.
- Every timing uses one CUDA event interval on one stream.

## Gates

1. All direct outputs are finite, deterministic and agree with the resident
   device output at normalized RMSE <= 0.001 and normalized max error <= 0.005.
2. Direct M8 complete-expert latency is <= 0.50 ms.
3. Direct M8 is no slower than staged M8.
4. Direct M1 is no slower than staged M1.

## Claim boundary

The result selects a cache-miss transport primitive for a single expert record.
It excludes disk reads, route lookup, cache policy, concurrent copies and
whole-layer route unions. A red speed gate is a valid falsification and does not
invalidate Phase49's resident route-adaptive kernels.

