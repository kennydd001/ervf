"""Y2-R1: does cutting expert bytes buy time? Measured without a per-call sync.

The first Y2 pass synchronised after every kernel call, so a ~7 us launch plus
sync sat on top of a ~40 us kernel and compressed the byte-scaling slope; the
curve came out non-monotone (75% slower than 100%, 25% slower than 50%), which
is the signature of a fixed cost dominating. This pass issues M calls back to
back and synchronises once, which is how S9's block-size probe measured.

Loads exactly one real expert record straight from the shard, so it needs no
routed bank and no decode loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
FRACTIONS = [1.0, 0.875, 0.75, 0.5, 0.25]
GATE_HALF = 0.40
CALLS = 200
ROUNDS = 9


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    import cupy as cp

    idx = ShardIndex(MODEL_DIR)
    cfg = idx.config
    hidden = cfg["hidden_size"]
    inter = cfg["moe_intermediate_size"]
    layer = cfg["layers_block_type"].index("moe")
    pre = f"backbone.layers.{layer}.mixer.experts.0.up_proj"

    codes_h = idx.read_raw(f"{pre}.weight")
    scales_h = idx.read_raw(f"{pre}.weight_scale")
    gscale = idx.get_scalar(f"{pre}.weight_scale_2")
    print(f"real record {pre}: codes {codes_h.nbytes:,} B, scales {scales_h.nbytes:,} B",
          flush=True)

    fused = FusedNVFP4()
    codes_full = cp.asarray(codes_h).reshape(inter, hidden // 2)
    scales_full = cp.asarray(scales_h).reshape(inter, hidden // 16)
    x = cp.asarray(np.random.default_rng(3).standard_normal(hidden).astype(np.float32))
    out = cp.zeros(inter, dtype=cp.float32)

    rows = {}
    for frac in FRACTIONS:
        cols = int(hidden * frac) // 16 * 16
        codes = cp.ascontiguousarray(codes_full[:, :cols // 2]).reshape(-1)
        scales = cp.ascontiguousarray(scales_full[:, :cols // 16]).reshape(-1)
        xs = cp.ascontiguousarray(x[:cols])
        nbytes = int(codes.nbytes + scales.nbytes)

        for _ in range(20):
            fused.gemv_into(out, codes, scales, xs, gscale, inter, cols)
        cp.cuda.Device(0).synchronize()

        per_call = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for _ in range(CALLS):
                fused.gemv_into(out, codes, scales, xs, gscale, inter, cols)
            cp.cuda.Device(0).synchronize()
            per_call.append((time.perf_counter_ns() - t0) / 1e3 / CALLS)
        us = float(np.percentile(per_call, 50))
        rows[f"{frac:.3f}"] = {
            "fraction": frac, "cols": cols, "bytes": nbytes,
            "us_per_call_p50": us, "us_raw": per_call,
            "gb_s": nbytes / (us * 1e-6) / 1e9,
        }
        print(f"  {frac:>6.1%} cols={cols:>5} bytes={nbytes:>9,} "
              f"{us:8.3f} us/call  {rows[f'{frac:.3f}']['gb_s']:6.1f} GB/s", flush=True)

    full = rows["1.000"]["us_per_call_p50"]
    half = rows["0.500"]["us_per_call_p50"]
    saving = 1.0 - half / full
    b_arr = np.array([r["bytes"] for r in rows.values()], dtype=np.float64)
    t_arr = np.array([r["us_per_call_p50"] for r in rows.values()], dtype=np.float64)
    slope, intercept = np.polyfit(b_arr, t_arr, 1)
    floor_share = intercept / full

    payload = {
        "kind": "lightningstream_nemotron_y2r1_bytes_vs_time",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "Y2_R1_BYTES_VS_TIME",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "tensor": pre,
        "config": {"rows": inter, "full_cols": hidden, "fractions": FRACTIONS,
                   "calls_per_round": CALLS, "rounds": ROUNDS,
                   "sync": "once per round, not per call"},
        "rows": rows,
        "halving_saving": saving,
        "linear_fit": {"us_per_byte": float(slope),
                       "intercept_us": float(intercept),
                       "intercept_share_of_full": float(floor_share)},
        "gate_G_Y2R1": {"required_halving_saving": GATE_HALF,
                        "measured": saving, "passed": bool(saving >= GATE_HALF)},
        "claim_boundary": (
            "Single-kernel microbenchmark of the real NVFP4 up_proj GEMV on one "
            "real expert record, with the column count reduced to shrink the "
            "record and the structure otherwise unchanged. It is a cost oracle "
            "for byte-reduction schemes: it says what a codec that removed those "
            "bytes could save on this kernel. It is NOT a quantizer, NOT a "
            "quality claim, and NOT a token time; it is not converted to tokens "
            "per second. The linear fit's intercept is the byte-independent part "
            "of this kernel's cost as measured, not a theoretical floor."),
    }
    (OUT_DIR / "y2r1_bytes_vs_time.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  halving the bytes saves {saving * 100:.1f}% (gate {GATE_HALF * 100:.0f}%) "
          f"-> {'PASS' if saving >= GATE_HALF else 'FAIL'}")
    print(f"  linear fit: {slope * 1e6:.4f} us per MB + {intercept:.2f} us fixed "
          f"({floor_share * 100:.1f}% of the full-record call)")
    print("\nwritten y2r1_bytes_vs_time.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
