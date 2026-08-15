"""N4-R1 runner: zero-cache routed path with the fused NVFP4 kernel.

Executes ``N4R1_FUSED_NVFP4_KERNEL_PREREGISTRATION_2026-08-14.md``.
Same working set and same host-bank layout as N4, so the comparison is like for
like.  CuPy throughout: NVRTC compiles the kernel, no host toolchain required.
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

from moe_lab.lightningstream_nemotron import reference as ref  # noqa: E402
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

SEED = 20260814
TOP_K = 6
MOE_LAYERS = [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38,
              40, 43, 45, 47, 49, 51]
HIDDEN = 2688
MOE_INTERMEDIATE = 1856
CODE_BYTES = 4_988_928
SCALE_BYTES = 623_616
GLOBAL_BYTES = 16
RECORD_BYTES = CODE_BYTES + SCALE_BYTES + GLOBAL_BYTES
HALF_CODE = CODE_BYTES // 2
HALF_SCALE = SCALE_BYTES // 2

WARMUP = 5
REPEATS = 30
GATE_P95_MS = 45.0
ARCH_STOP_MS = 60.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_compute_apps() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        rows = []
        for line in out.stdout.strip().splitlines():
            if line.strip():
                pid, used = [x.strip() for x in line.split(",")]
                rows.append({"pid": int(pid), "used_mib": int(used)})
        return rows
    except Exception:
        return [{"pid": -1, "used_mib": -1, "error": "nvidia-smi query failed"}]


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        used, free, temp = [x.strip() for x in out.stdout.strip().split(",")]
        return {"memory_used_mib": int(used), "memory_free_mib": int(free),
                "temperature_c": int(temp)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def percentiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()), "min": float(arr.min())}


def main() -> int:
    import cupy as cp

    started = utc_now()
    rng = np.random.default_rng(SEED)
    index = ShardIndex(MODEL_DIR)

    compute_apps = gpu_compute_apps()
    other = [a for a in compute_apps if a["pid"] != os.getpid()]
    gpu_before = gpu_state()
    if other or gpu_before.get("memory_used_mib", 0) > 256:
        print("BLOCKED: another PID holds a CUDA context; not interfering.")
        return 4

    # ------------------------------------------------------ routes and bank
    capture = json.loads((OUT_DIR / "n3_official_route_capture.json").read_text(encoding="utf-8"))
    layer1 = [int(e) for e in capture["indices"][0]]
    route_weights = [float(w) for w in capture["weights"][0]]

    routes = {}
    for layer in MOE_LAYERS:
        routes[layer] = layer1 if layer == 1 else sorted(
            rng.choice(128, size=TOP_K, replace=False).tolist())

    records = [(layer, expert) for layer in MOE_LAYERS for expert in routes[layer]]
    n_records = len(records)
    working_set = n_records * RECORD_BYTES

    # Pinned host bank, identical three-range layout to N4.
    h_codes_mem = cp.cuda.alloc_pinned_memory(n_records * CODE_BYTES)
    h_scales_mem = cp.cuda.alloc_pinned_memory(n_records * SCALE_BYTES)
    h_codes = np.frombuffer(h_codes_mem, dtype=np.uint8, count=n_records * CODE_BYTES)
    h_scales = np.frombuffer(h_scales_mem, dtype=np.uint8, count=n_records * SCALE_BYTES)
    globals_np = np.zeros((n_records, 4), dtype=np.float32)

    for slot, (layer, expert) in enumerate(records):
        prefix = f"backbone.layers.{layer}.mixer.experts.{expert}"
        code_parts, scale_parts = [], []
        for matrix in ("down_proj", "up_proj"):
            code_parts.append(index.read_raw(f"{prefix}.{matrix}.weight"))
            scale_parts.append(index.read_raw(f"{prefix}.{matrix}.weight_scale"))
        h_codes[slot * CODE_BYTES:(slot + 1) * CODE_BYTES] = np.concatenate(code_parts)
        h_scales[slot * SCALE_BYTES:(slot + 1) * SCALE_BYTES] = np.concatenate(scale_parts)
        globals_np[slot, 0] = index.get_scalar(f"{prefix}.down_proj.weight_scale_2")
        globals_np[slot, 1] = index.get_scalar(f"{prefix}.up_proj.weight_scale_2")

    d_codes = cp.empty(n_records * CODE_BYTES, dtype=cp.uint8)
    d_scales = cp.empty(n_records * SCALE_BYTES, dtype=cp.uint8)
    fused = FusedNVFP4()

    x_tok = cp.asarray(rng.standard_normal(HIDDEN) * 0.5, dtype=cp.float32)
    act = cp.zeros(MOE_INTERMEDIATE, dtype=cp.float32)
    tmp = cp.zeros(HIDDEN, dtype=cp.float32)
    acc = cp.zeros(HIDDEN, dtype=cp.float32)
    stream = cp.cuda.Stream.null

    def transfer_all():
        d_codes.set(h_codes)
        d_scales.set(h_scales)

    def compute_only():
        acc.fill(0)
        for slot in range(n_records):
            c0 = slot * CODE_BYTES
            s0 = slot * SCALE_BYTES
            fused.expert(
                d_codes[c0 + HALF_CODE:c0 + CODE_BYTES],
                d_scales[s0 + HALF_SCALE:s0 + SCALE_BYTES],
                float(globals_np[slot, 1]),
                d_codes[c0:c0 + HALF_CODE],
                d_scales[s0:s0 + HALF_SCALE],
                float(globals_np[slot, 0]),
                x_tok, act, tmp, HIDDEN, MOE_INTERMEDIATE)
            fused.accumulate_into(acc, tmp, route_weights[slot % TOP_K], HIDDEN)

    def composed():
        acc.fill(0)
        for layer_idx in range(len(MOE_LAYERS)):
            base = layer_idx * TOP_K
            c0, c1 = base * CODE_BYTES, (base + TOP_K) * CODE_BYTES
            s0, s1 = base * SCALE_BYTES, (base + TOP_K) * SCALE_BYTES
            d_codes[c0:c1].set(h_codes[c0:c1])
            d_scales[s0:s1].set(h_scales[s0:s1])
            for slot in range(base, base + TOP_K):
                cc = slot * CODE_BYTES
                ss = slot * SCALE_BYTES
                fused.expert(
                    d_codes[cc + HALF_CODE:cc + CODE_BYTES],
                    d_scales[ss + HALF_SCALE:ss + SCALE_BYTES],
                    float(globals_np[slot, 1]),
                    d_codes[cc:cc + HALF_CODE],
                    d_scales[ss:ss + HALF_SCALE],
                    float(globals_np[slot, 0]),
                    x_tok, act, tmp, HIDDEN, MOE_INTERMEDIATE)
                fused.accumulate_into(acc, tmp, route_weights[slot % TOP_K], HIDDEN)

    def timed(fn, warmup=WARMUP, repeats=REPEATS):
        for _ in range(warmup):
            fn()
            stream.synchronize()
        out = []
        for _ in range(repeats):
            stream.synchronize()
            t0 = time.perf_counter_ns()
            fn()
            stream.synchronize()
            out.append((time.perf_counter_ns() - t0) / 1e6)
        return out

    transfer_all()
    stream.synchronize()

    transport_raw = timed(transfer_all)
    compute_raw = timed(compute_only)
    composed_raw = timed(composed)

    transport_stats = percentiles(transport_raw)
    compute_stats = percentiles(compute_raw)
    composed_stats = percentiles(composed_raw)

    print(f"  transport only  p50 {transport_stats['p50']:8.3f} ms  p95 {transport_stats['p95']:8.3f} ms")
    print(f"  fused compute   p50 {compute_stats['p50']:8.3f} ms  p95 {compute_stats['p95']:8.3f} ms")
    print(f"  composed token  p50 {composed_stats['p50']:8.3f} ms  p95 {composed_stats['p95']:8.3f} ms")

    # ------------------------------------------------------------ correctness
    norm_w = index.get_float32("backbone.layers.1.norm.weight")
    hidden_np = (np.random.default_rng(SEED).standard_normal((1, HIDDEN)) * 0.5)
    x_ref = ref.rms_norm(hidden_np, norm_w, index.config["layer_norm_epsilon"])[0]
    probe = layer1[0]
    prefix = f"backbone.layers.1.mixer.experts.{probe}"
    up_w = index.dequantize_linear(f"{prefix}.up_proj")
    down_w = index.dequantize_linear(f"{prefix}.down_proj")
    expected_act = np.maximum(x_ref @ up_w.T, 0.0) ** 2
    expected_out = ref.mlp_relu2(x_ref[None, :], up_w, down_w)[0]

    x_c = cp.asarray(x_ref, dtype=cp.float32)
    a_c = cp.zeros(MOE_INTERMEDIATE, dtype=cp.float32)
    o_c = cp.zeros(HIDDEN, dtype=cp.float32)
    slot0 = records.index((1, probe))
    c0, s0 = slot0 * CODE_BYTES, slot0 * SCALE_BYTES
    fused.expert(
        d_codes[c0 + HALF_CODE:c0 + CODE_BYTES], d_scales[s0 + HALF_SCALE:s0 + SCALE_BYTES],
        float(globals_np[slot0, 1]),
        d_codes[c0:c0 + HALF_CODE], d_scales[s0:s0 + HALF_SCALE],
        float(globals_np[slot0, 0]),
        x_c, a_c, o_c, HIDDEN, MOE_INTERMEDIATE)
    stream.synchronize()
    got_act = cp.asnumpy(a_c).astype(np.float64)
    got_out = cp.asnumpy(o_c).astype(np.float64)
    rel_act = float(np.linalg.norm(got_act - expected_act) / np.linalg.norm(expected_act))
    rel_out = float(np.linalg.norm(got_out - expected_out) / np.linalg.norm(expected_out))
    all_finite = bool(np.isfinite(got_out).all() and np.isfinite(cp.asnumpy(acc)).all())

    pool = cp.get_default_memory_pool()
    peak_device = int(pool.total_bytes())
    gpu_after = gpu_state()

    gates = {
        "F1_expert_output_within_1e-5": rel_out <= 1e-5,
        "F2_activation_within_1e-5": rel_act <= 1e-5,
        "F3_composed_p95_under_45ms": composed_stats["p95"] <= GATE_P95_MS,
        "F4_peak_device_under_8gib": peak_device <= 8 * (1024 ** 3),
        "F5_no_dequantized_matrix_materialised": peak_device < working_set + 64 * (1024 ** 2),
        "F6_all_outputs_finite": all_finite,
    }

    result = {
        "kind": "lightningstream_nemotron_n4r1_fused_dataplane",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N4_R1_FUSED_NVFP4_EXPERT_KERNEL",
        "started_utc": started,
        "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "kernel_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "codec_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/nvfp4.py"),
        "cupy_version": cp.__version__,
        "compute_capability": cp.cuda.Device(0).compute_capability,
        "seed": SEED,
        "records_per_token": n_records,
        "record_bytes": RECORD_BYTES,
        "working_set_bytes": working_set,
        "warmup": WARMUP, "repeats": REPEATS,
        "non_interference": {"gpu_compute_apps": compute_apps,
                             "foreign": other, "gpu_before": gpu_before},
        "gpu_after": gpu_after,
        "transport_only": {"wall_ms": transport_stats, "raw_wall_ms": transport_raw},
        "fused_compute_only": {"wall_ms": compute_stats, "raw_wall_ms": compute_raw},
        "composed_token": {"wall_ms": composed_stats, "raw_wall_ms": composed_raw},
        "correctness": {
            "expert_output_rel_l2": rel_out,
            "activation_rel_l2": rel_act,
            "all_finite": all_finite,
            "reference": "N3-validated numpy reference",
            "bit_identity_claimed": False,
            "bit_identity_note": (
                "The fused kernel reduces with a block-parallel warp-shuffle tree, "
                "not the sequential order of the reference. Cross-backend bit "
                "identity is not demanded and is NOT claimed here."),
        },
        "peak_device_pool_bytes": peak_device,
        "baseline_n4": {
            "transport_p50_ms": 29.756,
            "unfused_decode_only_p50_ms": 353.133,
            "unfused_composed_p50_ms": 376.244,
            "unfused_composed_p95_ms": 403.649,
        },
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "architectural_stop_precondition": {
            "batched_transfer_tested": True,
            "correct_fused_kernel_present": True,
            "precondition_satisfied": True,
        },
        "architectural_stop_triggered": composed_stats["p95"] > ARCH_STOP_MS,
        "claim_boundary": (
            "Physically measured cache-free routed-expert path for one token on "
            "this specific GPU, reference-matching within a declared tolerance. "
            "NOT tokens per second, full-model latency, quality, memory "
            "feasibility of the complete runtime, bit-exactness of the fused "
            "output, kernel novelty, or any cross-runtime comparison. A component "
            "measurement is never promoted to tok/s."
        ),
    }
    result["terminal_state"] = (
        "n4r1_fused_zero_cache_screen_pass" if result["gates_all_pass"]
        else ("n4r1_architectural_stop" if result["architectural_stop_triggered"]
              else "n4r1_fused_screen_fail"))

    (OUT_DIR / "n4r1_fused_dataplane.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"expert rel_l2   : {rel_out:.3e}")
    print(f"activation      : {rel_act:.3e}")
    print(f"peak pool bytes : {peak_device:,}")
    speedup = 376.244 / composed_stats["p50"]
    print(f"composed speedup vs unfused N4 p50 : {speedup:.2f}x")
    for key, value in gates.items():
        print(f"  {'OK  ' if value else 'FAIL'} {key}")
    print(f"terminal state  : {result['terminal_state']}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
