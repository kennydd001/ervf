# Pure GPTQ fast-grid equivalence — attempt C lock

Locked after attempt B failed (164 code mismatches, 11 BF16 scale-bit mismatches, 1.138× speedup) and before opening attempt-C outputs.

Attempt C processes one expert and one projection at a time with the official tensor shapes. Hessian construction, BF16 Cholesky round-trip, inverse, block order, column order, hard quantize, 2-D error outer product, and block update remain sequential and unchanged. Only the 80 independent shrink candidates inside the pinned symmetric per-channel MSE scale search are stacked into a leading dimension. Each error reduction remains over the same contiguous 128 group columns; first-index `argmin` implements the official strict-improvement tie rule.

The same locked 10 experts and 30 matrices are tested. Pass requires zero integer-code and zero BF16 scale-bit mismatches. Failure forces the unmodified pinned name-neutral per-matrix implementation.
