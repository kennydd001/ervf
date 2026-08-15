# STREAMQ5-MoE P3A - integrated physical expert dataplane preregistration

Locked on 2026-08-12 after independent P2A kernel and P2C physical-H2D passes,
before P3A route or runtime output.

## Candidate

Use the verified 17.3671875-GiB pinned P1D bank, the physically resident
1,640-slot cache, and the selected 20-static plus 15/14-dynamic exact LRU.
The Q5 kernels consume cache records directly. Per layer and token they execute
all official top-8 experts: fused gate+up Q5 GEMV, FP32 SwiGLU, down Q5 GEMV,
and an eight-expert FP32 mean reduction into the next layer's 2,048-state.

All cache-miss copies and kernels run serially on the same CUDA stream. Future
layer routes may be trace-replayed to construct slot IDs, but no H2D/compute
overlap is credited; measured wall time therefore includes transfer, cache
bookkeeping, slot upload, 48 dependent expert layers, and launch scheduling.

## Fresh decision routes

Capture five new corrected-semantics 1,024-token routes, disjoint in aligned
128-token chunks from every prior decision route/input. Tokens 0-511 calibrate
static experts, 512-767 are validation, and 768-1023 are once-only test.

## Correctness and gates

One untimed fused one-layer smoke must match an independently decoded physical
reference at gate/up, SwiGLU, down, and reduced output (`max_abs <=0.02`,
`relative_l2 <=1e-4`). The unchanged evaluator then measures both splits.

Both validation and test require:

- aggregate and every-domain mean integrated expert wall time <=60 ms/token;
- aggregate and every-domain p95 integrated expert wall time <=75 ms/token;
- full pinned bank, cache+trunk+KV co-residency, >=384 MiB remaining scratch,
  exact miss simulation, finite outputs, and sampled transfer integrity;
- validation pass before once-only test.

P3A proves the complete routed-expert data plane, not attention, router/trunk
kernels, embeddings/head, KV update, sampling, or end-to-end model tokens/s.
