# PORT80B-D10A1-R independent component-result audit

**Verdict:** the raw result's formal negative is exactly reproduced: **18/21 gates true**, with failures in header-byte verification, GDN convolution state and shared Q5. None of the three is clean evidence of a mechanism failure: all are contaminated by cross-stream evaluator races. They cannot be promoted to passes, so endurance correctly remains closed.

## Recomputed result

- Validation wall p50/p95/p99: **78.771000 / 87.451660 / 90.157862 ms** (32 samples).
- CUDA-event p50/p95/p99: **78.738258 / 87.416735 / 90.118130 ms**.
- RAM after first touch: **3.680298 GiB**; validation endpoint loss **84.777 MiB**; maximum observed drawdown **97.879 MiB**.
- Minimum validation free VRAM: **2713.000 MiB**. Page-read maximum: **524.316/s**, but only 2 one-second samples exist.
- Raw error is null and unregister-failure list is empty. The raw JSON does not retain 48 individual register/unregister attempt rows, so their exact count is not independently reconstructable from raw arrays.

## The three failures

1. **Header gate â€” evaluator bug, transport not cleanly adjudicated.** Cases 12, 14, 15, 28 report 151, 156, 73, 111 mismatches (491 total). Yet all 40 actual header IDs/canaries equal their independent intended values, all 40 full Q5 candidate/oracle outputs are bitexact, and all 40 output digests are unique. `full_verify` creates its header reference and zero counter on the current/default stream, then launches on an explicit non-blocking stream without an event. These sporadic counts are therefore an invalid evaluator negative, not demonstrated corrupted records.

2. **GDN gate â€” evaluator bug, conv path unadjudicated.** The sampled recurrent result is exact (`max_abs=0`), but `conv_nonzero=0`. `conv.fill(0)` is launched on the default stream directly before the GDN kernel on the non-blocking stream. The asynchronous memset can overwrite the kernel's convolution-state writes.

3. **Shared-Q5 gate â€” evaluator bug, shared path unadjudicated.** Exactly **96,256 = 47Ã—2,048** elements differ, while layer 0 matches. A CPU byte audit shows all 48 shared records and the resident reference have the same complete codes+scales SHA-256 `79f02eb616ab264416a81032ceea5926c8d8ee55e18b1a685e0c551c3ae7772f`. The shared output buffers are zero-filled on the default stream immediately before shared kernels on the non-blocking stream, matching the observed layer-tail overwrite pattern.

The negative-control numerical outputs remain valid detections. Their header mismatch counts may also be polluted by the same verifier race; do not interpret the counts quantitatively.

## Protocol gaps retained

The `validation_32_finite` gate checks only wall-time finiteness, not composed state/output finiteness. The attention gate retains only absolute error despite its `abs_rel` name. The GDN reference compares only 4,096 recurrent cells and reduces convolution evidence to `count_nonzero`.

The stream-hygiene defect is broader than the three observed failures: pointer-table and route uploads, canary-error resets, dense-checksum reset, and the recurrent/conv/KV reset before validation also cross from the default stream to the explicit non-blocking stream without dependencies. Several passed gates have strong retained raw corroboration, but the composed validation state has no retained correctness digest and is therefore not semantically adjudicated. A repair must place allocations/uploads/fills inside the execution-stream context or use explicit CUDA-event waits, synchronize before host adjudication, and retain state/output finiteness plus digests and all 48 cleanup-attempt rows.

Conclusion: **no physical component mechanism failure is proven**, but neither are the three affected mechanisms proven passing. A new, preregistered evaluator must put allocations/fills and kernels on one stream or add explicit events, then rerun the component phase. No GPU work, registration, bank mutation or registry edit was performed by this audit.
