# S100 Phase 20R handoff

Do not use Nano assumptions.

The 12 unknown tensors from Phase20A are FP8 KV-cache scales. For scalar S:

    cache_code = FP8(K / S)
    K_approx    = decode(cache_code) * S

The existing runtime used S=1.

20R uses an algebraically equivalent correctness patch:

    K /= k_scale before cache write
    Q *= k_scale before QK
    V /= v_scale before cache write
    context *= v_scale after Score*V

This leaves the existing E4M3 cache representation intact. A later optimized
kernel can fuse these scalar operations.

The independent reference runs in a separate Transformers 5.14.1 environment,
with `use_mamba_kernels=False`. Config recognition alone is never enough to
open 20B.
