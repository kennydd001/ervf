# Official pure-GPTQ parallel fallback lock

Locked after accelerated attempts A, B, and C failed their zero-mismatch gates and before producing any full-bank codes.

Every matrix must now be quantized by the unmodified name-neutral pinned official per-matrix function `official_pure_gptq_projection`. Parallelism may exist only across independent OS processes and independent `(layer, expert)` tasks. No tensor batching, grid vectorization, shared Hessian, altered operation order, or changed CUDA stream is allowed inside a matrix call.

A pre-production benchmark may test 1, 2, 4, and 8 persistent worker processes. The selected worker count is the fastest configuration that remains under 7.5 GiB allocated VRAM and reproduces identical packed-code and BF16-scale hashes for a repeated control expert between solo and parallel execution. Each expert task writes an atomic, individually hashed artifact; layer assembly and full-bank verification remain separate.
