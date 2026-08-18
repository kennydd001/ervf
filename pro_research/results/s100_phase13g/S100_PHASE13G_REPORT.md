# S100 Phase 13G — lossless entropy codec

This test implements a real tile codec over representative resident FP8/BF16
streams. Each 1,024-byte tile uses a local top-symbol palette, fixed-width
palette/escape codes, and raw escape bytes. The decoder must reproduce every
original byte exactly.

This validates roundtrip correctness and exposes metadata/CPU overhead. It is
not yet a GPU inference decoder: no register/shared-memory decode kernel,
cuBLAS integration, or end-to-end latency/quality gate is included. Promotion
therefore remains closed.

The codec run used the first 1 MiB of each representative stream; Phase 13A
already supplied the full resident entropy census.

On six deterministic 1-MiB stream samples, every bit width roundtripped
exactly, but this simple palette codec expanded the data: mean encoded fraction
was 1.226 at 4 bits, 1.282 at 5 bits, and 1.303 at 6 bits. The Python decoder
managed only roughly 1.4–1.7 MiB/s. This closes this particular local-palette
encoding as a useful route; ANS/dictionary coding and a GPU decoder would be
a separate, untested hypothesis.
