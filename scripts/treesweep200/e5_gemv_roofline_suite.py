"""E5 — GEMV roofline recovery, weighted suite over every real NVFP4 shape.

Gates from the treesweep200 registry (frozen, not restated loosely):
  weighted_suite_bandwidth_ge_140_gb_s
  no_critical_shape_regression_gt_5pct
  strong: weighted_suite_bandwidth_ge_170_gb_s

NERVF-2 already measured 140.8 GB/s on the routed-expert up_proj alone. This runs
the suite the gate actually asks for: every shape gemv_nvfp4_rows is called with
in the real runtime, weighted by how often it is called per token, ERVF against
the production kernel, all arms L2-cold.
"""

from __future__ import annotations

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

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
GATE_SUITE = 140.0
GATE_STRONG = 170.0
GATE_REGRESSION = 0.05
CALLS = 100
ROUNDS = 7


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(v):
    return float(np.percentile(np.asarray(v, dtype=np.float64), 50))


def main() -> int:
    import cupy as cp

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    idx = ShardIndex(MODEL_DIR)
    cfg = idx.config
    hidden = cfg["hidden_size"]
    moe = [i for i, t in enumerate(cfg["layers_block_type"]) if t == "moe"]
    L0 = moe[0]
    n_moe, top_k = len(moe), cfg["num_experts_per_tok"]

    # Every shape gemv_nvfp4_rows is called with, and how often per token.
    suite = [
        ("routed_up", f"backbone.layers.{L0}.mixer.experts.0.up_proj", n_moe * top_k),
        ("shared_up", f"backbone.layers.{L0}.mixer.shared_experts.up_proj", n_moe),
        ("shared_down", f"backbone.layers.{L0}.mixer.shared_experts.down_proj", n_moe),
        ("lm_head", "lm_head", 1),
    ]

    fused = FusedNVFP4()
    l2 = int(cp.cuda.Device(0).attributes.get("L2CacheSize", 32 << 20))
    rng = np.random.default_rng(5)
    rows_out = {}

    for name, pre, calls_per_token in suite:
        ce = idx.entries[f"{pre}.weight"]
        rows, packed = ce.shape
        cols = packed * 2
        codes = cp.asarray(idx.read_raw(f"{pre}.weight"))
        scales = cp.asarray(idx.read_raw(f"{pre}.weight_scale"))
        gs = idx.get_scalar(f"{pre}.weight_scale_2")
        nbytes = int(codes.nbytes + scales.nbytes)
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        o1 = cp.zeros(rows, dtype=cp.float32)
        o2 = cp.zeros(rows, dtype=cp.float32)

        # L2-cold pool, as NERVF-1 established is necessary
        n_rep = max(2, (max(256 << 20, 8 * l2)) // nbytes)
        n_rep = min(n_rep, 96)
        pc = cp.tile(codes, n_rep)
        ps = cp.tile(scales, n_rep)
        cst, sst = int(codes.size), int(scales.size)
        ctr = {"i": 0}

        def rep():
            r = ctr["i"] % n_rep
            ctr["i"] += 1
            return pc[r * cst:(r + 1) * cst], ps[r * sst:(r + 1) * sst]

        def run(use):
            c_, s_ = rep()
            fused.use_ervf = use
            fused.gemv_into(o1 if not use else o2, c_, s_, x, gs, rows, cols)

        res = {}
        for label, use in (("base", False), ("ervf", True)):
            ctr["i"] = 0
            for _ in range(10):
                run(use)
            cp.cuda.Device(0).synchronize()
            per = []
            for _ in range(ROUNDS):
                t0 = time.perf_counter_ns()
                for _ in range(CALLS):
                    run(use)
                cp.cuda.Device(0).synchronize()
                per.append((time.perf_counter_ns() - t0) / 1e3 / CALLS)
            us = p50(per)
            res[label] = {"us": us, "gb_s": nbytes / (us * 1e-6) / 1e9, "raw": per}

        # exactness on this shape
        fused.use_ervf = False
        fused.gemv_into(o1, codes, scales, x, gs, rows, cols)
        fused.use_ervf = True
        fused.gemv_into(o2, codes, scales, x, gs, rows, cols)
        cp.cuda.Device(0).synchronize()
        exact = bool(cp.array_equal(o1, o2))
        fused.use_ervf = False

        rows_out[name] = {
            "tensor": pre, "rows": rows, "cols": cols, "bytes": nbytes,
            "calls_per_token": calls_per_token,
            "bytes_per_token": nbytes * calls_per_token,
            "base": res["base"], "ervf": res["ervf"],
            "speedup": res["base"]["us"] / res["ervf"]["us"],
            "bitwise_identical": exact,
        }
        r = rows_out[name]
        print("  %-12s %5dx%-5d base %8.2f us %6.1f GB/s | ervf %8.2f us %6.1f GB/s "
              "| %.3fx | exact=%s" % (name, rows, cols, res["base"]["us"],
                                      res["base"]["gb_s"], res["ervf"]["us"],
                                      res["ervf"]["gb_s"], r["speedup"], exact),
              flush=True)
        del pc, ps
        cp.get_default_memory_pool().free_all_blocks()

    # weighted by bytes actually moved per token
    tot_b = sum(r["bytes_per_token"] for r in rows_out.values())
    w_base = sum(r["base"]["us"] * r["calls_per_token"] for r in rows_out.values())
    w_ervf = sum(r["ervf"]["us"] * r["calls_per_token"] for r in rows_out.values())
    suite_base = tot_b / (w_base * 1e-6) / 1e9
    suite_ervf = tot_b / (w_ervf * 1e-6) / 1e9
    worst = min(r["speedup"] for r in rows_out.values())
    all_exact = all(r["bitwise_identical"] for r in rows_out.values())

    gates = {
        "weighted_suite_bandwidth_ge_140_gb_s": {
            "base_gb_s": suite_base, "ervf_gb_s": suite_ervf,
            "required": GATE_SUITE, "passed": bool(suite_ervf >= GATE_SUITE)},
        "no_critical_shape_regression_gt_5pct": {
            "worst_speedup": worst, "required_min": 1.0 - GATE_REGRESSION,
            "per_shape": {k: v["speedup"] for k, v in rows_out.items()},
            "passed": bool(worst >= 1.0 - GATE_REGRESSION)},
        "strong_weighted_suite_ge_170": {
            "ervf_gb_s": suite_ervf, "required": GATE_STRONG,
            "passed": bool(suite_ervf >= GATE_STRONG)},
        "exactness_all_shapes": {"passed": bool(all_exact)},
    }

    payload = {
        "kind": "treesweep200_e5_gemv_roofline_suite", "registry": "TREESWEEP200",
        "phase": "E5", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "fused_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "config": {"calls": CALLS, "rounds": ROUNDS, "l2_bytes": l2,
                   "moe_layers": n_moe, "top_k": top_k,
                   "mechanism": "ERVF (NERVF-2), subwarp width 16"},
        "shapes": rows_out,
        "weighted": {"bytes_per_token": tot_b, "base_gb_s": suite_base,
                     "ervf_gb_s": suite_ervf,
                     "speedup": w_base / w_ervf},
        "gates": gates,
        "claim_boundary": (
            "Single-kernel microbenchmarks on the real NVFP4 tensors of this "
            "checkpoint, every shape gemv_nvfp4_rows is called with in the real "
            "runtime, all arms cycling an L2-cold pool. The weighted figure "
            "weights each shape by its call count per token and is a KERNEL "
            "bandwidth, not a token time and not a throughput result. Exactness "
            "is checked per shape against the production kernel."),
    }
    (OUT_DIR / "E5_GEMV_ROOFLINE_SUITE.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n  weighted suite: base %.1f -> ervf %.1f GB/s (%.3fx)"
          % (suite_base, suite_ervf, w_base / w_ervf))
    print("  gate >=140: %s | strong >=170: %s | worst shape %.3fx | exact %s"
          % (suite_ervf >= GATE_SUITE, suite_ervf >= GATE_STRONG, worst, all_exact))
    print("\nwritten E5_GEMV_ROOFLINE_SUITE.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
