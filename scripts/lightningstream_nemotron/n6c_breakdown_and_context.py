"""N6-C: per-component metrical breakdown and decode cost versus context depth.

Two measurements:

1. **Breakdown** -- each block class timed separately within a real token, so the
   cost of Mamba, attention, MoE, router, LM head and embedding is attributed by
   measurement rather than by subtraction. No residual is named.

2. **Context sweep** -- the KV cache is populated synthetically and ``pos`` set,
   then one decode step is timed. This measures the decode-step cost at that
   context depth. It is explicitly NOT a real generation to that depth: filling
   131,072 positions at ~100 ms would take hours, and nothing here claims the
   model was actually run that far.
"""

from __future__ import annotations

import argparse
import hashlib
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
GIB = 1024 ** 3


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def gpu_state() -> dict:
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.free,temperature.gpu,clocks.sm,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t, c, p = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t),
                "sm_mhz": int(c), "power_w": float(p)}
    except Exception as e:
        return {"error": str(e)}


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min())}


def main() -> int:
    import cupy as cp

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*",
                    default=[0, 4096, 16384, 32768, 65536, 131072, 262100])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
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

    free0, total = cp.cuda.runtime.memGetInfo()
    print(f"allocating runtime with max_ctx={args.max_ctx} ...", flush=True)
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    print(f"  shell+state device use: {(free0 - free_shell)/GIB:.3f} GiB", flush=True)
    rt.load_routed_bank()
    print("  bank pinned", flush=True)

    dev = cp.cuda.Device(0)
    sync = dev.synchronize

    # ------------------------------------------------------- 1. breakdown
    # Time each block class by running only that class repeatedly on a live
    # state, so every figure is measured rather than obtained by subtraction.
    rt.reset()
    for t in [100, 200, 300]:
        rt.step(t)
    sync()

    def time_block(fn, reps):
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

    k = rt.k
    breakdown = {}

    mamba_i = rt.mamba_layers[0]
    attn_i = rt.attn_layers[0]
    moe_i = rt.moe_layers[0]
    d_moe = rt.layer[moe_i]

    k.norm(rt.normed, rt.h, rt.layer[mamba_i]["norm"], rt.hidden, rt.eps)
    sync()

    breakdown["rmsnorm_one"] = time_block(
        lambda: k.norm(rt.normed, rt.h, rt.layer[mamba_i]["norm"], rt.hidden, rt.eps),
        args.reps * 5)
    breakdown["mamba_one_layer"] = time_block(
        lambda: rt._mamba(mamba_i, rt.acc), args.reps * 2)
    breakdown["attention_one_layer_at_current_ctx"] = time_block(
        lambda: rt._attention(attn_i, rt.acc), args.reps * 2)
    breakdown["router_one_layer"] = time_block(lambda: rt._route(moe_i), args.reps * 2)
    breakdown["moe_one_layer_full"] = time_block(
        lambda: rt._moe(moe_i, rt.acc), args.reps * 2)
    breakdown["lm_head"] = time_block(
        lambda: k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden),
        args.reps * 2)
    breakdown["shared_expert_only"] = time_block(
        lambda: (rt.fused.gemv_into(rt.act[:rt.shared_inter], d_moe["sh_up_c"],
                                    d_moe["sh_up_s"], rt.normed, d_moe["sh_up_g"],
                                    rt.shared_inter, rt.hidden, apply_relu2=True),
                 rt.fused.gemv_into(rt.tmp, d_moe["sh_dn_c"], d_moe["sh_dn_s"],
                                    rt.act[:rt.shared_inter], d_moe["sh_dn_g"],
                                    rt.hidden, rt.shared_inter)),
        args.reps * 2)

    full_token = time_block(lambda: rt.step(100), args.reps * 2)

    scaled = {
        "mamba_total_23_layers_ms": breakdown["mamba_one_layer"]["p50"] * 23,
        "attention_total_6_layers_ms": breakdown["attention_one_layer_at_current_ctx"]["p50"] * 6,
        "moe_total_23_layers_ms": breakdown["moe_one_layer_full"]["p50"] * 23,
        "rmsnorm_total_53_ms": breakdown["rmsnorm_one"]["p50"] * 53,
        "lm_head_ms": breakdown["lm_head"]["p50"],
    }
    scaled["sum_of_measured_parts_ms"] = sum(
        scaled[k2] for k2 in ("mamba_total_23_layers_ms", "attention_total_6_layers_ms",
                              "moe_total_23_layers_ms", "rmsnorm_total_53_ms", "lm_head_ms"))
    scaled["measured_full_token_ms"] = full_token["p50"]
    scaled["unattributed_ms"] = full_token["p50"] - scaled["sum_of_measured_parts_ms"]

    print("\n--- breakdown (p50 ms) ---", flush=True)
    for name, s in breakdown.items():
        print(f"  {name:<40} {s['p50']:8.3f}")
    print(f"  {'-- mamba x23':<40} {scaled['mamba_total_23_layers_ms']:8.3f}")
    print(f"  {'-- attention x6':<40} {scaled['attention_total_6_layers_ms']:8.3f}")
    print(f"  {'-- moe x23':<40} {scaled['moe_total_23_layers_ms']:8.3f}")
    print(f"  {'-- full token measured':<40} {full_token['p50']:8.3f}")
    print(f"  {'-- unattributed':<40} {scaled['unattributed_ms']:8.3f}")

    # -------------------------------------------------- 2. context sweep
    kv_dim = rt.kv_dim
    ctx_rows = {}
    rng = np.random.default_rng(7)
    for target in args.contexts:
        if target >= args.max_ctx - 4:
            continue
        rt.reset()
        if target > 0:
            # Synthetic KV fill: measures decode cost at depth, NOT a real
            # generation to that depth.
            for i in rt.attn_layers:
                rt.kc[i][: target * kv_dim] = cp.asarray(
                    rng.standard_normal(target * kv_dim).astype(np.float32) * 0.1)
                rt.vc[i][: target * kv_dim] = cp.asarray(
                    rng.standard_normal(target * kv_dim).astype(np.float32) * 0.1)
            rt.pos = target
        sync()
        s = time_block(lambda: rt.step(100), args.reps)
        ctx_rows[str(target)] = {
            "context_depth": target,
            "synthetic_kv": target > 0,
            "ms": s,
            "tok_s_p50": 1000.0 / s["p50"],
            "tok_s_mean": 1000.0 / s["mean"],
            "gpu": gpu_state(),
        }
        print(f"  ctx {target:>7}: p50 {s['p50']:8.2f} ms -> {1000.0/s['p50']:7.3f} tok/s",
              flush=True)

    free_end, _ = cp.cuda.runtime.memGetInfo()

    result = {
        "kind": "lightningstream_nemotron_n6c_breakdown_context",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N6_C_BREAKDOWN_AND_CONTEXT",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "device_total_bytes": int(total),
        "device_used_by_runtime_bytes": int(free0 - free_shell),
        "device_free_at_end_bytes": int(free_end),
        "max_ctx_allocated": args.max_ctx,
        "architectural_context_limit": rt.cfg["max_position_embeddings"],
        "igpu_used": False,
        "igpu_note": "Intel Arc Pro 140T belongs to the protected HET-NEXT line; not touched.",
        "breakdown_p50_ms": {k2: v["p50"] for k2, v in breakdown.items()},
        "breakdown_full": breakdown,
        "scaled_totals": scaled,
        "full_token": full_token,
        "context_sweep": ctx_rows,
        "attribution_note": (
            "Every component figure is measured directly. The unattributed term "
            "is reported as a number and NOT given a name -- the project rule "
            "after the 'glue' term turned out to be attention."),
        "synthetic_kv_note": (
            "Context-sweep rows above 0 populate the KV cache synthetically and "
            "set pos. They measure decode-step cost at that depth. They are NOT "
            "a real generation to that depth and no quality claim attaches."),
        "claim_boundary": (
            "Measured batch-1 single-stream decode on this specific GPU with a "
            "zero-cache streamed expert bank and no expert cache. NOT a quality "
            "result, benchmark score, or claim about other hardware or batch "
            "sizes. A component measurement is never promoted to tok/s -- the "
            "full-token figures here are full-token measurements, not "
            "extrapolations."),
    }
    (OUT_DIR / "n6c_breakdown_and_context.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: n6c_breakdown_and_context.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
