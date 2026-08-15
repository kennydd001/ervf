"""S7 smoke: grouped GQA kernel vs current warp-per-position kernel.

C2-shape check on synthetic-but-realistic FP8 KV: same buffers through both
kernels, rel_l2 of the combined outputs; then per-layer timing at t=262,144.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    import cupy as cp
    from moe_lab.lightningstream_nemotron.gpu_kernels import GPUKernels

    k = GPUKernels()
    t, head_dim, n_heads, groups = 262144, 128, 32, 16
    n_kv = n_heads // groups
    max_ctx = 262144
    rng = np.random.default_rng(5)
    # realistic-ish: E4M3 codes uniform, q gaussian
    Kc = cp.asarray(rng.integers(0, 256, n_kv * max_ctx * head_dim, dtype=np.uint8))
    Vc = cp.asarray(rng.integers(0, 256, n_kv * max_ctx * head_dim, dtype=np.uint8))
    q = cp.asarray(rng.standard_normal(n_heads * head_dim).astype(np.float32) * 0.5)
    splits = 256
    part_acc = cp.zeros(n_heads * splits * 4 * head_dim, dtype=cp.float32)
    part_ml = cp.zeros(n_heads * splits * 4 * 2, dtype=cp.float32)
    out_old = cp.zeros(n_heads * head_dim, dtype=cp.float32)
    out_new = cp.zeros(n_heads * head_dim, dtype=cp.float32)
    scale = 1.0 / float(np.sqrt(head_dim))

    k.attention_fp8(out_old, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx,
                    scale, part_acc, part_ml)
    k.attention_fp8_gqa(out_new, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx,
                        scale, part_acc, part_ml)
    cp.cuda.Device(0).synchronize()
    a, b = cp.asnumpy(out_old), cp.asnumpy(out_new)
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    print(f"grouped vs current: rel_l2 {rel:.3e} max|d| {np.abs(a-b).max():.3e}")

    for name, fn in (("current", k.attention_fp8), ("grouped", k.attention_fp8_gqa)):
        for _ in range(3):
            fn(out_old, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx, scale,
               part_acc, part_ml)
        cp.cuda.Device(0).synchronize()
        ts = []
        for _ in range(15):
            t0 = time.perf_counter_ns()
            fn(out_old, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx, scale,
               part_acc, part_ml)
            cp.cuda.Device(0).synchronize()
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        print(f"{name}: p50 {np.percentile(ts, 50):.3f} ms/layer @262144")

    ok = rel < 1e-6
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
