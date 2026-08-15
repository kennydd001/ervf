"""S6: per-component breakdown at ctx 262100 (and ctx 0 control) on the
masked S5 runtime, with the cache enabled and warmed.

Diagnostic census only -- preregistered in S6_BREAKDOWN_262K_PREREGISTRATION.
Follows n6c's attribution discipline: every component measured directly, the
unattributed term reported as a number and never named.
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

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"


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

    started = datetime.now(timezone.utc).isoformat()
    rt = LightningRuntime(MODEL_DIR, contexts_max=262144, verbose=False)
    rt.enable_cache(31)
    rt.load_routed_bank()
    dev = cp.cuda.Device(0)
    sync = dev.synchronize
    k = rt.k
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]

    def time_block(fn, reps=20):
        for _ in range(3):
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

    results = {}
    for target in (0, 262100):
        rt.reset()
        if target:
            for j in range(64):
                rt.step(varied[j % len(varied)])
            rt.pos = target
            for j in range(32):           # warm cache at depth
                rt.step(varied[(j + 64) % len(varied)])
            sync()

        mamba_i, attn_i, moe_i = rt.mamba_layers[0], rt.attn_layers[0], rt.moe_layers[0]
        d_moe = rt.layer[moe_i]
        k.norm(rt.normed, rt.h, rt.layer[mamba_i]["norm"], rt.hidden, rt.eps)
        sync()

        bd = {}
        bd["rmsnorm_one"] = time_block(
            lambda: k.norm(rt.normed, rt.h, rt.layer[mamba_i]["norm"], rt.hidden, rt.eps))
        bd["mamba_one_layer"] = time_block(lambda: rt._mamba(mamba_i, rt.acc))
        bd["attention_one_layer_at_depth"] = time_block(lambda: rt._attention(attn_i, rt.acc))
        bd["router_one_layer"] = time_block(lambda: rt._route(moe_i))
        bd["moe_one_layer_hitpath"] = time_block(lambda: rt._moe(moe_i, rt.acc))
        # mixed path: different token each rep so routes vary naturally
        cnt = {"i": 0}
        def moe_mixed():
            cnt["i"] += 1
            rt.step(varied[(cnt["i"] + 128) % len(varied)])  # full step keeps state legal
        # full-token doubles as the mixed-MoE carrier; MoE-only mixed is not
        # measurable without a full step (routes depend on the hidden state)
        bd["lm_head"] = time_block(
            lambda: k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden))
        bd["shared_expert_only"] = time_block(
            lambda: (rt.fused.gemv_into(rt.act[:rt.shared_inter], d_moe["sh_up_c"],
                                        d_moe["sh_up_s"], rt.normed, d_moe["sh_up_g"],
                                        rt.shared_inter, rt.hidden, apply_relu2=True),
                     rt.fused.gemv_into(rt.tmp, d_moe["sh_dn_c"], d_moe["sh_dn_s"],
                                        rt.act[:rt.shared_inter], d_moe["sh_dn_g"],
                                        rt.hidden, rt.shared_inter)))
        full = time_block(lambda: rt.step(100), reps=30)

        scaled = {
            "mamba_x23": bd["mamba_one_layer"]["p50"] * 23,
            "attention_x6": bd["attention_one_layer_at_depth"]["p50"] * 6,
            "moe_hitpath_x23": bd["moe_one_layer_hitpath"]["p50"] * 23,
            "rmsnorm_x53": bd["rmsnorm_one"]["p50"] * 53,
            "lm_head": bd["lm_head"]["p50"],
        }
        scaled["sum_parts"] = sum(scaled.values())
        scaled["full_token"] = full["p50"]
        scaled["unattributed"] = full["p50"] - scaled["sum_parts"]
        results[str(target)] = {"breakdown_p50_ms": {k2: v["p50"] for k2, v in bd.items()},
                                "full": full, "scaled": scaled,
                                "cache": dict(rt.cache_stats)}
        print(f"\nctx {target}:")
        for name, s in bd.items():
            print(f"  {name:<34} {s['p50']:8.3f} ms")
        print(f"  {'full token':<34} {full['p50']:8.3f} ms "
              f"({1000/full['p50']:.2f} tok/s)  unattributed {scaled['unattributed']:.3f}",
              flush=True)

    out = {
        "kind": "lightningstream_nemotron_s6_breakdown_262k",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S6_BREAKDOWN_262K",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": "reports/lightningstream_nemotron/S6_BREAKDOWN_262K_PREREGISTRATION_2026-08-14.md",
        "results": results,
        "synthetic_kv_note": ("ctx 262100 rows use a populated cache and set pos; "
                              "decode-step cost at depth, not real generation to depth."),
        "claim_boundary": ("Cost attribution per component on this runtime/GPU/depth. "
                           "No tok/s targets, no quality claims, no projections."),
    }
    (OUT_DIR / "s6_breakdown_262k.json").write_text(json.dumps(out, indent=2) + "\n",
                                                    encoding="utf-8")
    print("\nwritten s6_breakdown_262k.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
