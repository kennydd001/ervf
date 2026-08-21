# S100 Phase61: Ornith two-warp route-bucket preregistration

Phase60's indirect M4 was exact and 1.143x faster than assignment-major M1, but
missed its frozen 1.15x gate. Resource audit showed 64 registers/thread.

The Phase61 candidate maps two physical warps to each output row. Each warp owns
four of the original eight virtual reduction partitions, halving per-thread
accumulator storage. The eight partition sums are written to shared memory and
combined in the original Phase60 order. Cache slots, input IDs, real weights,
four launches, controls, warm-up, repetitions and route patterns are unchanged.

Gates:

1. M1-M4 are bit-identical to Phase60's one-warp kernels and repeat exactly.
2. All outputs are finite.
3. Every kernel uses no local memory and at most 56 registers/thread.
4. Pair-warp M4 is at least 5% faster than the Phase60 one-warp M4 in the same
   process.
5. Pair-warp M4 is at least 1.15x faster than assignment-major M1.

This remains a hot, planned route-bucket primitive, not an end-to-end claim.
