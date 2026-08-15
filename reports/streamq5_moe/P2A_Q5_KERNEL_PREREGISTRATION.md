# STREAMQ5-MoE P2A - physical Q5 GEMV preregistration

Locked on 2026-08-12 after the independently verified P1D full-bank pass and
before the first STREAMQ5 Q5-kernel timing output.

## Fixed kernel and cases

The CUDA 13.2/NVRTC kernel consumes the physical P1D 5-bit payload and raw
BF16 scales directly. It may not materialize a dequantized matrix. There is
one CUDA block per output row, exactly 256 threads, FP32 accumulation, and the
physical decode rule `signed_code * float(BF16_scale)`.

Cases are layers 0, 24, and 47. For each layer, use the official eight route
IDs for the first P1C general-domain test token (token 768), and benchmark its
gate, up, and down matrices: exactly 72 records. Deterministic PCG64 FP32 input
vectors and all expert IDs are hash-locked before output.

One untimed toolchain/correctness smoke on the first case is allowed. The same
locked kernel must then be used unchanged for the benchmark. Each record uses
100 warm-ups and 500 measured launches. CUDA events report p50/p95/p99. A BF16
dequantized `torch.mv` is measured only as a baseline and is not the candidate.

## Correctness and performance gates

- all 72 physical-record outputs finite and within `max_abs <=0.02` and
  `relative_l2 <=1e-4` of an independent decoded FP32 reference;
- physical headers/CRC and source/result hashes valid;
- candidate source contains no full-matrix dequantization;
- aggregate p50 and conservative summed-p95 throughput each >=27.2 billion
  weight-applications/s;
- corresponding full-token routed-expert p95 compute projection <=66.615 ms
  for 1,811,939,328 expert weight-applications.

P2A proves physical expert-GEMV correctness and microkernel throughput only.
Pinned full-bank residency, fragmented H2D, attention/trunk execution, launch
scheduling across a token, and end-to-end tokens/s remain separate.
