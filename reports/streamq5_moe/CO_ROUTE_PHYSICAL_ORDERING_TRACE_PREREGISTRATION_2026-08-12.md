# Co-route-aware physical expert ordering — trace preregistration

Date: 2026-08-12

## Claim boundary

This is a CPU-only trace-stage test on the existing P4D Qwen30 top-8 route
captures. It changes only the hypothetical physical order of expert records;
routes, weights, router scores, cache state and arithmetic remain unchanged.
It does not build a bank, copy bytes, execute CUDA, measure latency, or prove
anything about Qwen3-Coder-Next-80B.

The P4D capture has exactly 1,024 tokens per domain. The fixed partitions are:

- learn: `[0, 512)`;
- validation: `[512, 768)`;
- test: `[768, 1024)`.

These partitions are strictly disjoint. They are not globally fresh:
TierFlow-F0 previously reported validation/test traffic metrics on the latter
two windows. No co-route physical ordering or interval-cover metric has yet
been computed. The result is therefore a preregistered test of a new
hypothesis on reused traces, not a fresh confirmatory dataset.

## Locked inputs

- capture manifest: `reports/streamq5_moe/p4d_route_capture_result.json`,
  expected SHA-256
  `7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb`;
- route tensors: `reports/runs/streamq5_moe/p4d_routes/layer_00.safetensors`
  through `layer_47.safetensors`, hashes locked by the capture manifest;
- 48 layers, five domains, 1,024 tokens/domain, top-8 of 128 experts;
- expert-record size: 3,035,136 bytes from the P4D input lock.

## Fixed learned ordering

One independent physical order is learned per layer using only `[0, 512)`
from all five domains.

1. Build a symmetric `128 x 128` integer co-occurrence matrix. Every unordered
   expert pair in a natural top-8 route adds one. Expert frequency is the
   number of natural route occurrences.
2. Process experts in descending frequency, tie by lower expert ID.
3. Start the order with the first expert. Insert each following expert into
   the position maximizing the change in adjacent co-occurrence weight:
   `w(left,e) + w(e,right) - w(left,right)`, omitting missing endpoint terms.
   Ties use the lowest insertion position.
4. The resulting 128 IDs must be an exact permutation. No validation or test
   value may alter it.

The identity expert-ID order is a descriptive baseline, not a selection arm.

## Fixed interval cover

For each natural top-8 route, map its expert IDs to positions in the learned
order and sort those eight positions. A transferred interval is inclusive and
contains every physical record between its endpoints.

The primary cover is the exact dynamic-programming optimum that:

1. covers all eight required expert records;
2. transfers at most one non-required record for that route;
3. minimizes interval count;
4. then minimizes transferred record count;
5. then chooses the lexicographically smallest interval list.

Thus any individual route transfers at most nine records (`1.125x` payload).
Reported payload inflation is total transferred records divided by exact
required records. Coverage errors are fatal controls.

## Validation gate and test opening

Validation passes only if the learned ordering simultaneously has:

1. aggregate p95 interval count `<= 2.0` per token/layer;
2. aggregate mean interval count `<= 1.5`;
3. per-domain p95 interval count `<= 3.0` for every domain;
4. aggregate payload inflation `<= 1.10x`;
5. exact coverage, valid permutations and learn-only provenance.

If any validation gate fails, test remains closed and no GPU or physical-bank
work is authorized. If validation passes, exactly the frozen learned orders
are evaluated once on `[768, 1024)` with the same gates. The test is not used
to change the ordering, interval budget, algorithm or thresholds.

## Stop/go

- Validation fail: record a negative trace result; no test and no GPU.
- Validation pass, test fail: record a held-out negative; no GPU.
- Validation and test pass: report the trace pass before any GPU work. A
  separate preregistration would then be required for a physical relaid-out
  bank and AB/BA copy benchmark.

Even a trace pass would establish only exact interval locality on Qwen30
routes. It would not establish a latency gain or transfer to a 512-expert 80B
router.
