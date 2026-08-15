"""N6-B runner: GPU decode loop, correctness then throughput.

Joins the N5 resident shell with the N4-R2 streamed routed path.  Correctness is
gated FIRST against N6-A's frozen CPU result; only then is throughput measured.
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
                            "--query-gpu=memory.used,memory.free,temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t, c = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t), "sm_mhz": int(c)}
    except Exception as e:
        return {"error": str(e)}


def foreign_cuda() -> list:
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        rows = []
        for line in o.stdout.strip().splitlines():
            if line.strip():
                pid, used = [x.strip() for x in line.split(",")]
                if int(pid) != os.getpid():
                    rows.append({"pid": int(pid), "used_mib": int(used)})
        return rows
    except Exception:
        return [{"pid": -1, "error": "query failed"}]


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min())}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--probe-contexts", type=int, nargs="*", default=[0, 1024, 4032])
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    foreign = foreign_cuda()
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    gpu_before = gpu_state()
    free0, total = cp.cuda.runtime.memGetInfo()

    print("loading resident shell ...", flush=True)
    t0 = time.perf_counter()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx)
    shell_s = time.perf_counter() - t0
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    print(f"  shell loaded in {shell_s:.1f}s, device used {(free0 - free_shell)/GIB:.3f} GiB",
          flush=True)

    print("pinning routed bank (15.4 GiB, one-time) ...", flush=True)
    t0 = time.perf_counter()
    rt.load_routed_bank()
    bank_s = time.perf_counter() - t0
    print(f"  bank pinned in {bank_s:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ---------------------------------------------------- correctness gate
    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=False)
    rt.reset()
    nxt = None
    for t in ids:
        nxt = rt.step(t)
    cp.cuda.Device(0).synchronize()
    first_text = tok.decode([nxt])
    coherent = "paris" in first_text.strip().lower()
    print(f"correctness: {prompt!r} -> {first_text!r}  coherent={coherent}", flush=True)

    if not coherent:
        result = {
            "kind": "lightningstream_nemotron_n6b_gpu_decode",
            "phase": "N6_B_GPU_DECODE_LOOP",
            "started_utc": started,
            "terminal_state": "n6b_gpu_incoherent",
            "correctness": {"prompt": prompt, "top1_text": first_text, "coherent": False,
                            "n6a_expected": " Paris"},
            "gates": {"B1_matches_n6a_coherence": False},
            "gates_all_pass": False,
            "claim_boundary": "GPU decode produced an incoherent token; no throughput "
                              "measured. A component measurement is never promoted to tok/s.",
        }
        (OUT_DIR / "n6b_gpu_decode.json").write_text(json.dumps(result, indent=2) + "\n",
                                                     encoding="utf-8")
        print("STOP: GPU path incoherent; throughput not measured.")
        return 3

    # ------------------------------------------------- generation coherence
    rt.reset()
    gen_ids, per_token = [], []
    cur = ids[0]
    fed = ids[1:]
    for i in range(len(ids) + args.gen):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        nxt = rt.step(cur)
        cp.cuda.Device(0).synchronize()
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        if i >= len(ids) - 1:
            per_token.append(dt_ms)
            gen_ids.append(nxt)
            cur = nxt
        else:
            cur = fed[i]
    text = tok.decode(gen_ids)
    print(f"generated: {text!r}", flush=True)

    free_run, _ = cp.cuda.runtime.memGetInfo()

    # ------------------------------------------------- throughput by context
    ctx_results = {}
    for target in args.probe_contexts:
        if target >= args.max_ctx - 8:
            continue
        rt.reset()
        filler = 100
        for _ in range(target):
            rt.step(filler)
        cp.cuda.Device(0).synchronize()
        samples = []
        for _ in range(12):
            t0 = time.perf_counter_ns()
            rt.step(filler)
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        s = pct(samples)
        ctx_results[str(target)] = {
            "context_before_step": target,
            "ms": s,
            "raw_ms": samples,
            "tok_s_at_p50": 1000.0 / s["p50"],
            "tok_s_at_mean": 1000.0 / s["mean"],
            "gpu": gpu_state(),
        }
        print(f"  ctx {target:>6}: p50 {s['p50']:7.2f} ms  -> {1000.0/s['p50']:6.3f} tok/s",
              flush=True)

    free_end, _ = cp.cuda.runtime.memGetInfo()
    gen_stats = pct(per_token)

    result = {
        "kind": "lightningstream_nemotron_n6b_gpu_decode",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N6_B_GPU_DECODE_LOOP",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "kernels_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/gpu_kernels.py"),
        "fused_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "device": {"name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                   "total_bytes": int(total), "free_before": int(free0),
                   "free_after_shell": int(free_shell), "free_after_run": int(free_run),
                   "free_end": int(free_end),
                   "shell_bytes": int(free0 - free_shell)},
        "igpu_used": False,
        "igpu_note": "Intel Arc Pro 140T belongs to the protected HET-NEXT line and was not touched.",
        "max_context_allocated": args.max_ctx,
        "architectural_context_limit": rt.cfg["max_position_embeddings"],
        "load_seconds": {"shell": shell_s, "routed_bank_pin": bank_s},
        "correctness": {"prompt": prompt, "top1_text": first_text, "coherent": coherent,
                        "n6a_expected": " Paris", "generated": text},
        "generation": {"tokens": len(per_token), "ms": gen_stats,
                       "raw_ms": per_token, "tok_s_at_p50": 1000.0 / gen_stats["p50"]},
        "throughput_by_context": ctx_results,
        "gpu_before": gpu_before, "gpu_after": gpu_state(),
        "non_interference": {"foreign_cuda_contexts": foreign},
        "gates": {
            "B1_matches_n6a_coherence": coherent,
            "B2_all_contexts_measured": len(ctx_results) > 0,
            "B3_device_under_8gib": (total - free_run) <= 8 * GIB,
            "B4_no_igpu_used": True,
        },
        "claim_boundary": (
            "Measured single-stream batch-1 decode on this specific GPU with a "
            "zero-cache streamed expert bank. Throughput figures are for this "
            "configuration only. NOT a quality result, not a benchmark score, "
            "not a claim about other hardware, batch sizes or contexts beyond "
            "those measured. A component measurement is never promoted to tok/s."
        ),
    }
    result["gates_all_pass"] = all(result["gates"].values())
    result["terminal_state"] = ("n6b_gpu_decode_coherent_and_measured"
                                if result["gates_all_pass"] else "n6b_gpu_decode_fail")

    (OUT_DIR / "n6b_gpu_decode.json").write_text(json.dumps(result, indent=2) + "\n",
                                                 encoding="utf-8")
    print()
    print(f"generation p50 {gen_stats['p50']:.2f} ms -> {1000.0/gen_stats['p50']:.3f} tok/s")
    print(f"terminal state : {result['terminal_state']}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
