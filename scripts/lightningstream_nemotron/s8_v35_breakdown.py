"""S8: component breakdown of the v35 runtime, to explain the GQA gap.

Kimi's S7 mechanism check measured 10.66x KV amplification at kernel level
(heads=32 5.8463 ms vs heads=2 0.5483 ms at 244.79 GB/s, the device roofline).
The grouped kernel is wired, yet end-to-end delivers far less than that factor
predicts. This attributes the token by direct measurement rather than by
subtraction, so the gap is either explained or relocated.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT = REPO / "reports" / "lightningstream_nemotron" / "s8_v35_breakdown.json"
DEPTHS = [0, 262100]
REPS = 12


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


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

    rt = LightningRuntime(MODEL, contexts_max=262144, embed_on_host=True)
    rt.enable_cache(72)
    rt.load_routed_bank()
    sync = cp.cuda.Device(0).synchronize

    def timed(fn, reps=REPS, warm=3):
        for _ in range(warm):
            fn()
        sync()
        out = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter_ns()
            fn()
            sync()
            out.append((time.perf_counter_ns() - t0) / 1e6)
        return pct(out)

    k = rt.k
    mamba_i, attn_i, moe_i = rt.mamba_layers[0], rt.attn_layers[0], rt.moe_layers[0]
    d_moe = rt.layer[moe_i]
    rng = np.random.default_rng(5)
    varied = [int(v) for v in rng.integers(1000, 60000, size=512)]

    results = {}
    for depth in DEPTHS:
        rt.reset()
        for j in range(min(depth, 48)):
            rt.step(varied[j % len(varied)])
        rt.pos = depth
        for j in range(32):
            rt.step(varied[(j + 48) % len(varied)])
        sync()

        k.norm(rt.normed, rt.h, rt.layer[mamba_i]["norm"], rt.hidden, rt.eps)
        sync()

        b = {
            "rmsnorm_one": timed(lambda: k.norm(rt.normed, rt.h,
                                                rt.layer[mamba_i]["norm"],
                                                rt.hidden, rt.eps), REPS * 3),
            "mamba_one_layer": timed(lambda: rt._mamba(mamba_i, rt.acc)),
            "attention_one_layer": timed(lambda: rt._attention(attn_i, rt.acc)),
            "router_one_layer": timed(lambda: rt._route(moe_i)),
            "moe_one_layer": timed(lambda: rt._moe(moe_i, rt.acc)),
            "lm_head": timed(lambda: (
                rt.fused.gemv_into(rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                                   rt.normed, rt.lm_head_g, rt.vocab, rt.hidden)
                if rt.lm_head_kind == "nvfp4" else
                k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden))),
        }
        full = timed(lambda: rt.step(varied[0]), REPS)

        scaled = {
            "mamba_x23": b["mamba_one_layer"]["p50"] * 23,
            "attention_x6": b["attention_one_layer"]["p50"] * 6,
            "moe_x23": b["moe_one_layer"]["p50"] * 23,
            "rmsnorm_x53": b["rmsnorm_one"]["p50"] * 53,
            "lm_head": b["lm_head"]["p50"],
        }
        scaled["sum_parts"] = sum(scaled.values())
        scaled["full_token"] = full["p50"]
        scaled["unattributed"] = full["p50"] - scaled["sum_parts"]

        results[str(depth)] = {"per_call_p50_ms": {n: v["p50"] for n, v in b.items()},
                               "full": full, "scaled": scaled,
                               "cache": dict(rt.cache_stats)}
        print(f"\n--- depth {depth} ---")
        for n, v in b.items():
            print(f"  {n:<22} {v['p50']:8.3f} ms")
        for n, v in scaled.items():
            print(f"  {'* ' + n:<22} {v:8.3f} ms")

    payload = {
        "kind": "lightningstream_nemotron_s8_v35_breakdown",
        "phase": "S8_V35_BREAKDOWN",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL.name,
        "capacity": 72,
        "results": results,
        "s7_mechanism_reference": {
            "heads_32_ms": 5.8463, "heads_2_ms": 0.5483, "ratio": 10.66,
            "note": "kernel-level amplification measured by Kimi on synthetic KV",
        },
        "claim_boundary": (
            "Component attribution on this runtime, GPU and depth. Every figure "
            "is measured directly, never by subtraction; the unattributed term "
            "is reported and deliberately not named. No tok/s target claim."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
