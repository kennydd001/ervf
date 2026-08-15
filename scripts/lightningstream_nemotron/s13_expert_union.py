"""S13: expert-union over speculative windows, measured on real greedy routes.

Preregistration: S13_EXPERT_UNION_PREREGISTRATION_2026-08-15.md (frozen before
this run).  Builds nothing: decode with the existing step(capture_routes=...),
then count |union of routes| over non-overlapping W-token windows per MoE layer.

Gates:
  G-S13-C1  generated tokens bit-identical to s10a_mtp_acceptance.json sequences
  G-S13-S1  W=1 union == 6.0 everywhere; pooled mean non-decreasing in W
  G-S13-U1  pooled mean union at W=5 <= 12.0  ->  S10 step 2 (build) not refuted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3
WINDOWS = [2, 3, 4, 5, 6, 8]
GATE_U1_W = 5
GATE_U1_MAX = 12.0


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def gpu_state():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.free,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t, p = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t), "power_w": float(p)}
    except Exception as e:
        return {"error": str(e)}


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "min": float(a.min())}


def union_stats(routes_by_layer: dict[str, list[list[int]]], steps: int):
    """Non-overlapping W-windows over the step axis, per layer, then pooled."""
    per_w = {w: [] for w in WINDOWS}
    per_layer_w5 = {}
    for layer, rows in routes_by_layer.items():
        arr = np.asarray(rows[:steps], dtype=np.int32)  # [steps, k]
        layer_w5 = []
        for w in WINDOWS:
            n_win = arr.shape[0] // w
            for j in range(n_win):
                u = int(np.unique(arr[j * w:(j + 1) * w, :]).size)
                per_w[w].append(u)
                if w == GATE_U1_W:
                    layer_w5.append(u)
        per_layer_w5[layer] = float(np.mean(layer_w5)) if layer_w5 else None
    return {"per_window": {str(w): pct(per_w[w]) for w in WINDOWS},
            "per_layer_mean_w5": per_layer_w5}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--steps-gate", type=int, default=124)
    ap.add_argument("--steps-4k", type=int, default=64)
    ap.add_argument("--ctx-4k", type=int, default=4096)
    args = ap.parse_args()

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

    corpus_path = OUT_DIR / "s10a_corpus.json"
    accept_path = OUT_DIR / "s10a_mtp_acceptance.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    accept = json.loads(accept_path.read_text(encoding="utf-8"))

    started = datetime.now(timezone.utc).isoformat()
    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    cache_bytes = rt.enable_cache(args.capacity)
    free_cache, _ = cp.cuda.runtime.memGetInfo()
    rt.load_routed_bank()
    print(f"shell {(free0-free_shell)/GIB:.3f} GiB | cache {cache_bytes/GIB:.3f} GiB | "
          f"free {free_cache/GIB:.3f} GiB", flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    arms = []
    c1_rows = []

    def run_arm(label: str, prompt_ids: list[int], steps: int):
        rt.reset()
        cap: dict[str, list] = {}
        for t in prompt_ids[:-1]:
            rt.step(t)  # teacher-forced prefill, no capture
        # feed the last prompt token, then decode `steps` tokens with capture
        cur = rt.step(prompt_ids[-1], capture_routes=cap)
        gen = [cur]
        for _ in range(steps - 1):
            cur = rt.step(cur, capture_routes=cap)
            gen.append(cur)
        cp.cuda.Device(0).synchronize()
        stats = union_stats(cap, steps)
        print(f"arm {label}: steps={steps} layers={len(cap)} "
              f"union@W5 mean={stats['per_window']['5']['mean']:.3f} "
              f"p95={stats['per_window']['5']['p95']:.1f}", flush=True)
        arms.append({"label": label, "prompt_tokens": len(prompt_ids),
                     "steps": steps, "generated": gen,
                     "routes": {k: v for k, v in cap.items()},
                     "union": stats, "gpu": gpu_state()})
        return gen

    # ---- Arm A: the three frozen S10A gate prompts
    for gp, ref in zip(corpus["gate_prompts"], accept["gate_prompts"]):
        ids = tok.encode(gp["text"], add_special_tokens=False)
        gen = run_arm(f"A-{ref['label']}", ids, args.steps_gate)
        expected = ref["sequence"][ref["prompt_tokens"]:
                                   ref["prompt_tokens"] + args.steps_gate]
        ok = gen == expected
        first_diff = next((i for i, (a, b) in enumerate(zip(gen, expected))
                           if a != b), None)
        c1_rows.append({"arm": f"A-{ref['label']}", "identical": ok,
                        "compared_tokens": len(expected), "first_diff": first_diff})
        print(f"  C1 A-{ref['label']}: identical={ok}", flush=True)

    # ---- Arm B: 4K natural-text prompt from the same corpus
    ids4k = tok.encode(corpus["long_ctx_text"], add_special_tokens=False)[:args.ctx_4k]
    run_arm("B-4k", ids4k, args.steps_4k)

    # ---- Gates
    c1_pass = all(r["identical"] for r in c1_rows)

    s1_w1_ok, mono_ok = True, True
    pooled_w = {w: [] for w in WINDOWS}
    for arm in arms:
        means = [arm["union"]["per_window"][str(w)]["mean"] for w in WINDOWS]
        for w, m in zip(WINDOWS, means):
            pooled_w[w].append(m)
        if any(means[i] > means[i + 1] for i in range(len(means) - 1)):
            mono_ok = False
    # W=1 sanity: every captured route row must hold exactly 6 unique expert ids.
    for arm in arms:
        for rows in arm["routes"].values():
            for row in rows:
                if len(set(row)) != 6:
                    s1_w1_ok = False
    grand = {str(w): float(np.mean(pooled_w[w])) for w in WINDOWS}
    if any(grand[str(WINDOWS[i])] > grand[str(WINDOWS[i + 1])] + 1e-12
           for i in range(len(WINDOWS) - 1)):
        mono_ok = False
    s1_pass = s1_w1_ok and mono_ok

    u1_value = grand[str(GATE_U1_W)]
    u1_pass = u1_value <= GATE_U1_MAX

    free_end, _ = cp.cuda.runtime.memGetInfo()
    payload = {
        "kind": "lightningstream_nemotron_s13_expert_union",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S13_EXPERT_UNION",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src" / "moe_lab" /
                                      "lightningstream_nemotron" / "runtime.py"),
        "corpus_sha256": sha256_path(corpus_path),
        "acceptance_sha256": sha256_path(accept_path),
        "config": {"capacity_per_layer": args.capacity, "max_ctx": args.max_ctx,
                   "embed_on_host": True, "backbone_fp8_kv": True,
                   "decode": "greedy_argmax", "windows": WINDOWS,
                   "windowing": "non_overlapping", "top_k": 6,
                   "n_routed_experts": 128,
                   "steps_gate": args.steps_gate, "steps_4k": args.steps_4k,
                   "ctx_4k_prompt": args.ctx_4k},
        "memory": {"shell_gib": (free0 - free_shell) / GIB,
                   "cache_gib": cache_bytes / GIB,
                   "free_end_gib": free_end / GIB},
        "cache_stats": rt.cache_stats,
        "arms": arms,
        "c1_rows": c1_rows,
        "pooled_mean_by_window": grand,
        "gates": {
            "G_S13_C1": {"requirement": "generated == s10a sequences, 3 gate prompts",
                         "pass": c1_pass, "rows": c1_rows},
            "G_S13_S1": {"requirement": "W=1 union == 6 everywhere; pooled mean "
                                        "non-decreasing in W",
                         "w1_exactly_6": s1_w1_ok, "monotone": mono_ok,
                         "pass": s1_pass},
            "G_S13_U1": {"requirement": f"pooled mean union at W={GATE_U1_W} "
                                        f"<= {GATE_U1_MAX}",
                         "measured": u1_value, "pass": u1_pass},
        },
        "gpu": gpu_state(),
        "claim_boundary": (
            "Routing statistic on real greedy generation at short and 4K context "
            "on this checkpoint and this runtime. No speculative loop was built, "
            "no throughput was measured, 262K was not measured (a real 262K "
            "prefill costs ~2.6 h of sequential steps on this decode-only "
            "runtime; route context-dependence is secondary per S10A's stable A "
            "between ctx ~200 and 4096). The G-S13-U1 verdict decides only "
            "whether MoE bytes per verification sweep refute building a "
            "speculative loop; it says nothing about acceptance, draft cost, or "
            "final tokens/s. No quality claim, no benchmark score, no statement "
            "about other hardware, batch sizes, or prompts."),
    }
    out = OUT_DIR / "s13_expert_union.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"G-S13-C1 {'PASS' if c1_pass else 'FAIL'}  "
          f"G-S13-S1 {'PASS' if s1_pass else 'FAIL'}  "
          f"G-S13-U1 {'PASS' if u1_pass else 'FAIL'} (union@W5={u1_value:.3f})")
    return 0 if (c1_pass and s1_pass) else 2


if __name__ == "__main__":
    sys.exit(main())
