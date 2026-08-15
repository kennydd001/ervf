"""S9-B: is the expert GEMV reduction/occupancy bound rather than bandwidth bound?

S9-A killed the launch hypothesis (13% of the MoE term). The up-GEMV runs 1856
blocks that each read only 1344 B, so each block does ~1.3 vector loads per
thread at blockDim 256 and then pays a full warp+shared reduction tree. If that
is the cost, a smaller block should be faster: fewer warps to reduce across,
more work per thread, better occupancy per SM.

Pure component microbenchmark on one real expert. No tok/s claim.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT = REPO / "reports" / "lightningstream_nemotron" / "s9b_blocksize_probe.json"
HIDDEN, INTER = 2688, 1856
BLOCKS = [32, 64, 128, 256, 512]


def main() -> int:
    import cupy as cp

    idx = ShardIndex(MODEL)
    pre = "backbone.layers.1.mixer.experts.0"
    up_c = cp.asarray(idx.read_raw(f"{pre}.up_proj.weight"))
    up_s = cp.asarray(idx.read_raw(f"{pre}.up_proj.weight_scale"))
    up_g = idx.get_scalar(f"{pre}.up_proj.weight_scale_2")

    rng = np.random.default_rng(11)
    x = cp.asarray(rng.standard_normal(HIDDEN).astype(np.float32))
    out = cp.zeros(INTER, dtype=cp.float32)
    sync = cp.cuda.Device(0).synchronize

    ref = None
    rows = []
    for b in BLOCKS:
        f = FusedNVFP4(block=b)
        for _ in range(20):
            f.gemv_into(out, up_c, up_s, x, up_g, INTER, HIDDEN, apply_relu2=True)
        sync()
        got = cp.asnumpy(out).copy()
        if ref is None:
            ref = got
            rel = 0.0
        else:
            rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))

        N = 300
        t0 = time.perf_counter_ns()
        for _ in range(N):
            f.gemv_into(out, up_c, up_s, x, up_g, INTER, HIDDEN, apply_relu2=True)
        sync()
        us = (time.perf_counter_ns() - t0) / 1e3 / N
        # 2,494,464 B of codes + 311,808 B of scales per up_proj
        gbs = (2_494_464 + 311_808) / (us / 1e6) / 1e9
        rows.append({"block": b, "us": us, "eff_gb_s": gbs, "rel_l2_vs_block32": rel})
        print(f"  block {b:>4}: {us:8.2f} us  {gbs:7.1f} GB/s  rel_l2 {rel:.2e}")

    best = min(rows, key=lambda r: r["us"])
    cur = next(r for r in rows if r["block"] == 256)
    res = {
        "kind": "lightningstream_nemotron_s9b_blocksize_probe",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": f"{pre}.up_proj [{INTER},{HIDDEN}] NVFP4",
        "bytes_read_per_call": 2_494_464 + 311_808,
        "rows": rows,
        "current_block": 256,
        "current_us": cur["us"],
        "best_block": best["block"],
        "best_us": best["us"],
        "speedup_vs_current": cur["us"] / best["us"],
        "outputs_identical_across_blocks": all(r["rel_l2_vs_block32"] == 0.0 for r in rows),
        "claim_boundary": ("Single-matrix component microbenchmark on this GPU. "
                           "Not a token measurement and not a tok/s claim; a "
                           "runtime change would need its own gated run."),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\n  best block {best['block']} -> {cur['us'] / best['us']:.2f}x vs current 256")
    print(f"  outputs identical across block sizes: {res['outputs_identical_across_blocks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
