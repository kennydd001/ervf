"""S7 step 1: GQA amplification mechanism check.

Times attn_decode_warp_fp8 at t=262,144 with grid heads=32 (current) vs
heads=2 (one q-head per kv group, same splits/chunk). Prediction if HBM
traffic is amplified per q-head: ~16x time ratio. Component measurement only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"


def main() -> int:
    import cupy as cp

    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines()
                   if l.strip() and int(l.split(",")[0]) != os.getpid()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    from moe_lab.lightningstream_nemotron.gpu_kernels import GPUKernels
    k = GPUKernels()

    t, head_dim, n_kv, groups = 262144, 128, 2, 16
    max_ctx = 262144
    rng = np.random.default_rng(3)
    Kc = cp.asarray(rng.integers(0, 256, n_kv * max_ctx * head_dim, dtype=np.uint8))
    Vc = cp.asarray(rng.integers(0, 256, n_kv * max_ctx * head_dim, dtype=np.uint8))
    q = cp.asarray(rng.standard_normal(32 * head_dim).astype(np.float32))
    splits = max(1, min(k.MAX_SPLITS, (t + k.SPLIT_THRESHOLD - 1) // k.SPLIT_THRESHOLD))
    chunk = (t + splits - 1) // splits
    part_acc = cp.zeros(32 * splits * 4 * head_dim, dtype=cp.float32)
    part_ml = cp.zeros(32 * splits * 4 * 2, dtype=cp.float32)

    res = {}
    for heads in (32, 2):
        fn = lambda: k.attn_decode_warp_fp8(
            (heads, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups if heads == 32 else 1), np.int32(max_ctx),
             np.float32(1.0 / np.sqrt(head_dim)), np.int32(chunk)))
        for _ in range(3):
            fn()
        cp.cuda.Device(0).synchronize()
        times = []
        for _ in range(15):
            t0 = time.perf_counter_ns()
            fn()
            cp.cuda.Device(0).synchronize()
            times.append((time.perf_counter_ns() - t0) / 1e6)
        res[f"heads_{heads}_ms_p50"] = float(np.percentile(times, 50))
        print(f"heads={heads}: p50 {res[f'heads_{heads}_ms_p50']:.3f} ms", flush=True)

    ratio = res["heads_32_ms_p50"] / res["heads_2_ms_p50"]
    kv_bytes = n_kv * t * head_dim * 2
    print(f"ratio 32/2 = {ratio:.2f}x  (16x = full amplification)")
    print(f"heads=2 effective KV bandwidth: {kv_bytes/res['heads_2_ms_p50']/1e6:.1f} GB/s")
    out = {
        "kind": "lightningstream_nemotron_s7_mechanism_check",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        **res, "ratio_32_over_2": ratio,
        "kv_bytes_per_layer": kv_bytes,
        "heads2_effective_kv_gbs": kv_bytes / res["heads_2_ms_p50"] / 1e6,
        "amplification_hypothesis_supported": bool(ratio >= 4.0),
        "claim_boundary": "Component kernel timing on synthetic KV; not tok/s.",
    }
    (OUT_DIR / "s7_mechanism_check.json").write_text(json.dumps(out, indent=2) + "\n",
                                                     encoding="utf-8")
    print("written s7_mechanism_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
