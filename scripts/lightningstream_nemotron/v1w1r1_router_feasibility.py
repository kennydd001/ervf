"""V1 / W1-R1: is a device-side router feasible, and does W1's gain exist at depth?

Preregistered in V1W1R1_PREREGISTRATION_2026-08-15.md.

V1     A device-side router can only avoid the host on a cache miss by having the
       GEMV read the missing record straight from mapped pinned host memory
       through the same UVA pointer gather_down_sparse already uses. This times
       that read against the device one, on one real expert record.

W1-R1  W1 measured +0.511 ms at 262100 against 4.520 ms of arm-to-arm drift. Same
       gate, tighter design: base/fast/base triplets on consecutive steps in the
       same warm state, so neighbouring samples share their thermal state.
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

from moe_lab.lightningstream_nemotron.runtime import (  # noqa: E402
    LightningRuntime, UP_CODE, UP_SCALE)

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 64
MISS_RATE = 0.1785            # K0 LRU replay at capacity 72
EXPERT_CALLS = 138            # 23 MoE layers x top_k 6
Y1_SYNC_MS = 6.656            # Y1, ctx 262100
V1_BUDGET_US = Y1_SYNC_MS * 1000.0 / (MISS_RATE * EXPERT_CALLS)
CALLS = 200
ROUNDS = 9
TRIPLETS = 24
GATE_DRIFT_MS = 0.5
GATE_SIGN = 0.60


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pctl(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def host_view(cp, host_array, offset, nbytes):
    """A cupy array over pinned host memory, addressable from the device by UVA."""
    ptr = host_array.ctypes.data + offset
    mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=host_array)
    return cp.ndarray((nbytes,), dtype=cp.uint8,
                      memptr=cp.cuda.MemoryPointer(mem, 0))


def generate(rt, cp, tokenizer):
    out = []
    for text in PROMPTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        gen = [int(nxt)]
        for _ in range(GEN_TOKENS - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append(gen)
    return out


def triplet_sweep(rt, cp, target, max_ctx, triplets):
    """base/fast/base on consecutive steps, same warm state."""
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=8192)]
    rt.fast_host = False
    rt.reset()
    for j in range(min(target, 64)):
        rt.step(varied[j % len(varied)])
    rt.pos = target
    for j in range(32):
        rt.step(varied[(j + 64) % len(varied)])
    cp.cuda.Device(0).synchronize()

    rows, k = [], 96
    for _ in range(triplets):
        vals = []
        for fast in (False, True, False):
            rt.fast_host = fast
            t0 = time.perf_counter_ns()
            rt.step(varied[k % len(varied)])
            cp.cuda.Device(0).synchronize()
            vals.append((time.perf_counter_ns() - t0) / 1e6)
            k += 1
        b1, f, b2 = vals
        rows.append({"b1": b1, "fast": f, "b2": b2,
                     "effect_ms": 0.5 * (b1 + b2) - f,
                     "drift_ms": abs(b2 - b1)})
    rt.fast_host = False
    return rows


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 262100])
    ap.add_argument("--triplets", type=int, default=TRIPLETS)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ------------------------------------------------------------------- V1
    print("\nV1: mapped-host GEMV against device GEMV", flush=True)
    layer0 = rt.moe_layers[0]
    bank = rt.bank[layer0]
    hidden, inter = rt.hidden, rt.moe_inter
    e = 0
    dev_codes = cp.asarray(bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE])
    dev_scales = cp.asarray(bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE])
    hst_codes = host_view(cp, bank["up_codes"], e * UP_CODE, UP_CODE)
    hst_scales = host_view(cp, bank["up_scales"], e * UP_SCALE, UP_SCALE)
    gs = bank["g_up"][e]
    x = cp.asarray(np.random.default_rng(5).standard_normal(hidden).astype(np.float32))
    out_d = cp.zeros(inter, dtype=cp.float32)
    out_h = cp.zeros(inter, dtype=cp.float32)

    rt.fused.gemv_into(out_d, dev_codes, dev_scales, x, gs, inter, hidden)
    rt.fused.gemv_into(out_h, hst_codes, hst_scales, x, gs, inter, hidden)
    cp.cuda.Device(0).synchronize()
    identical = bool(cp.array_equal(out_d, out_h))
    print(f"  bit-identical device vs mapped host: {identical}", flush=True)

    v1 = {}
    for label, c_, s_ in (("device", dev_codes, dev_scales),
                          ("mapped_host", hst_codes, hst_scales)):
        for _ in range(20):
            rt.fused.gemv_into(out_d, c_, s_, x, gs, inter, hidden)
        cp.cuda.Device(0).synchronize()
        per = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for _ in range(CALLS):
                rt.fused.gemv_into(out_d, c_, s_, x, gs, inter, hidden)
            cp.cuda.Device(0).synchronize()
            per.append((time.perf_counter_ns() - t0) / 1e3 / CALLS)
        us = float(np.percentile(per, 50))
        v1[label] = {"us_per_call_p50": us, "us_raw": per,
                     "gb_s": (UP_CODE + UP_SCALE) / (us * 1e-6) / 1e9}
        print(f"  {label:<12} {us:8.2f} us/call  {v1[label]['gb_s']:6.1f} GB/s",
              flush=True)

    delta_us = v1["mapped_host"]["us_per_call_p50"] - v1["device"]["us_per_call_p50"]
    feasible = identical and delta_us < V1_BUDGET_US
    print(f"  extra per miss {delta_us:.2f} us against a budget of "
          f"{V1_BUDGET_US:.2f} us -> feasible={feasible}", flush=True)

    # ---------------------------------------------------------------- W1-R1
    print("\nW1-R1: base/fast/base triplets", flush=True)
    rt.fast_host = False
    ref = generate(rt, cp, tokenizer)
    rt.fast_host = True
    alt = generate(rt, cp, tokenizer)
    rt.fast_host = False
    ident_gen = ref == alt
    print(f"  generation bit-identical: {ident_gen}", flush=True)

    w1r1 = {}
    for ctx in args.contexts:
        if ctx >= args.max_ctx - 8:
            continue
        rows = triplet_sweep(rt, cp, ctx, args.max_ctx, args.triplets)
        eff = [r["effect_ms"] for r in rows]
        dr = [r["drift_ms"] for r in rows]
        wins = sum(1 for r in rows if r["effect_ms"] > 0)
        w1r1[str(ctx)] = {
            "context": ctx, "triplets": len(rows), "rows": rows,
            "effect": pctl(eff), "drift": pctl(dr),
            "sign_fraction": wins / len(rows),
            "resolution_ok": bool(float(np.percentile(dr, 50)) < GATE_DRIFT_MS),
            "sign_ok": bool(float(np.percentile(eff, 50)) > 0
                            and wins / len(rows) >= GATE_SIGN),
        }
        r = w1r1[str(ctx)]
        print(f"  ctx {ctx:>6}: effect p50 {r['effect']['p50']:+.3f} ms | "
              f"drift p50 {r['drift']['p50']:.3f} ms | sign {r['sign_fraction']:.2f} | "
              f"resolution_ok={r['resolution_ok']} sign_ok={r['sign_ok']}", flush=True)

    deep = str(args.contexts[-1])
    gates = {
        "G_V1_C1_exactness": {"passed": bool(identical)},
        "G_V1_F1_feasible": {
            "budget_us_per_miss": V1_BUDGET_US, "measured_delta_us": delta_us,
            "miss_rate": MISS_RATE, "expert_calls": EXPERT_CALLS,
            "y1_sync_ms": Y1_SYNC_MS, "passed": bool(feasible)},
        "G_W1R_C1_identity": {"passed": bool(ident_gen)},
        "G_W1R_R1_resolution": {c: v["resolution_ok"] for c, v in w1r1.items()},
        "G_W1R_E1_sign": {c: v["sign_ok"] for c, v in w1r1.items()},
        "G_W1_P1_unchanged": {
            "required_gain_ms_at_deep": 1.0,
            "measured_effect_p50_ms": w1r1[deep]["effect"]["p50"] if deep in w1r1 else None,
            "passed": bool(deep in w1r1 and w1r1[deep]["effect"]["p50"] >= 1.0)},
    }

    payload = {
        "kind": "lightningstream_nemotron_v1w1r1_router_feasibility",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "V1_W1R1",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "triplets": args.triplets,
                   "calls_per_round": CALLS, "rounds": ROUNDS},
        "v1": {"bit_identical": bool(identical), "arms": v1,
               "delta_us_per_miss": delta_us, "budget_us_per_miss": V1_BUDGET_US,
               "record_bytes": UP_CODE + UP_SCALE},
        "w1r1": w1r1,
        "gates": gates,
        "claim_boundary": (
            "V1 times the same NVFP4 up_proj GEMV kernel on the same bytes, once "
            "from device memory and once through a UVA pointer into the pinned "
            "host bank. It is a feasibility number for a device-side router that "
            "would service misses from host, NOT a router and NOT a token time; "
            "the budget it is compared against is arithmetic on Y1's measured "
            "sync cost and K0's measured miss rate. W1-R1 is a paired design: "
            "each triplet's effect is the fast step against the mean of its two "
            "neighbouring base steps, which removes a thermal trend but not "
            "per-step noise. The adoption gate G-W1-P1 is unchanged at 1.0 ms."),
    }
    (OUT_DIR / "v1w1r1_router_feasibility.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-V1-C1 exactness  : {gates['G_V1_C1_exactness']['passed']}")
    print(f"  G-V1-F1 feasible   : {feasible}")
    print(f"  G-W1R-C1 identity  : {ident_gen}")
    for c, v in w1r1.items():
        print(f"  ctx {c:>6} resolution={v['resolution_ok']} sign={v['sign_ok']}")
    print(f"  G-W1-P1 (unchanged): {gates['G_W1_P1_unchanged']['passed']}")
    print("\nwritten v1w1r1_router_feasibility.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
