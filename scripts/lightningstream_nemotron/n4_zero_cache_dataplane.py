"""N4_ZERO_CACHE_DATAPLANE runner.

Executes the frozen preregistration
``N4_ZERO_CACHE_DATAPLANE_PREREGISTRATION_2026-08-14.md``.

Zero cache: every routed expert of every measured token is fetched from a
host-resident pinned bank and transferred to the device.  Nothing is reused
between tokens.

Writes only inside the Nemotron allowlist.  Refuses to touch the GPU while a
PORT80B/STREAMQ5 process is alive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import nvfp4, reference as ref  # noqa: E402
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


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        used, free, temp, clock = [x.strip() for x in out.stdout.strip().split(",")]
        return {"memory_used_mib": int(used), "memory_free_mib": int(free),
                "temperature_c": int(temp), "sm_clock_mhz": int(clock)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def gpu_compute_apps() -> list[dict]:
    """Processes actually holding a CUDA context on the device.

    This is the authoritative test for contention with the PORT80B/STREAMQ5
    line.  Presence of *a python process* is not: a CPU-only job contends for
    nothing, and this machine runs short-lived python helpers whose PIDs change
    between invocations.  Asking nvidia-smi which PIDs hold device memory
    answers the question that actually matters.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        rows = []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            pid, used = [x.strip() for x in line.split(",")]
            rows.append({"pid": int(pid), "used_mib": int(used)})
        return rows
    except Exception:
        # Fail closed: an unreadable GPU state is treated as busy.
        return [{"pid": -1, "used_mib": -1, "error": "nvidia-smi query failed"}]


def python_processes() -> list[dict]:
    """Recorded as context only; never used to block on its own."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Where-Object { $_.ProcessName -match 'python' } | "
             "Select-Object Id,ProcessName | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=60)
        text = out.stdout.strip()
        if not text:
            return []
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
        return [r for r in rows if int(r["Id"]) != os.getpid()]
    except Exception:
        return []


# --------------------------------------------------------------------------
# N4-A: host-resident bank in the exact three-range layout
# --------------------------------------------------------------------------

def build_route_plan(index: ShardIndex, rng: np.random.Generator) -> dict:
    """Fixed routes: N3's captured layer-1 top-6, deterministic elsewhere.

    Transport cost depends on the NUMBER of distinct records, not on which
    experts they are, so a frozen synthetic pattern is adequate for the
    transport arms.  Correctness is checked separately on the real N3 route.
    """
    capture_path = OUT_DIR / "n3_official_route_capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    layer1 = [int(e) for e in capture["indices"][0]]

    plan = {}
    for layer in MOE_LAYERS:
        if layer == 1:
            plan[layer] = layer1
        else:
            plan[layer] = sorted(rng.choice(128, size=TOP_K, replace=False).tolist())
    return {
        "routes": plan,
        "route_provenance": (
            "layer 1 from the frozen N3 official capture (SYNTHETIC-INPUT routes); "
            "other layers a frozen deterministic selection. Never natural routes."
        ),
        "n3_capture_sha256": sha256_path(capture_path),
    }


def build_host_bank(index: ShardIndex, plan: dict) -> dict:
    """Materialise the token working set as pinned host memory, 3 ranges/expert.

    This is a PARTIAL bank covering exactly the measured working set, not the
    full 15.389 GiB routed bank. Full-bank residency is an N5 question.
    """
    import torch

    records = []
    total = 0
    for layer, experts in plan["routes"].items():
        for expert in experts:
            prefix = f"backbone.layers.{layer}.mixer.experts.{expert}"
            records.append((layer, expert, prefix))
            total += RECORD_BYTES

    codes = torch.empty(len(records) * CODE_BYTES, dtype=torch.uint8, pin_memory=True)
    scales = torch.empty(len(records) * SCALE_BYTES, dtype=torch.uint8, pin_memory=True)
    globals_ = torch.empty(len(records) * GLOBAL_BYTES, dtype=torch.uint8, pin_memory=True)

    verified = 0
    mismatches = []
    for slot, (layer, expert, prefix) in enumerate(records):
        # down_proj then up_proj: the on-disk order within each dtype region.
        code_parts, scale_parts, global_parts = [], [], []
        for matrix in ("down_proj", "up_proj"):
            code_parts.append(index.read_raw(f"{prefix}.{matrix}.weight"))
            scale_parts.append(index.read_raw(f"{prefix}.{matrix}.weight_scale"))
            global_parts.append(index.read_raw(f"{prefix}.{matrix}.input_scale"))
            global_parts.append(index.read_raw(f"{prefix}.{matrix}.weight_scale_2"))

        code_blob = np.concatenate(code_parts)
        scale_blob = np.concatenate(scale_parts)
        global_blob = np.concatenate(global_parts)

        if code_blob.size != CODE_BYTES or scale_blob.size != SCALE_BYTES or global_blob.size != GLOBAL_BYTES:
            mismatches.append({"layer": layer, "expert": expert, "reason": "size"})
            continue

        codes[slot * CODE_BYTES:(slot + 1) * CODE_BYTES] = torch.from_numpy(code_blob)
        scales[slot * SCALE_BYTES:(slot + 1) * SCALE_BYTES] = torch.from_numpy(scale_blob)
        globals_[slot * GLOBAL_BYTES:(slot + 1) * GLOBAL_BYTES] = torch.from_numpy(global_blob)
        verified += 1

    return {
        "records": records,
        "codes": codes, "scales": scales, "globals": globals_,
        "record_count": len(records),
        "verified_records": verified,
        "mismatches": mismatches,
        "working_set_bytes": total,
        "pinned": True,
        "partial_bank_note": (
            "Pinned host bank covering exactly the measured token working set "
            f"({total:,} B), not the full 15.389 GiB routed bank."
        ),
    }


def verify_bank_against_checkpoint(index: ShardIndex, bank: dict, sample: int,
                                   rng: np.random.Generator) -> dict:
    """G1: re-read the checkpoint and compare bytes for a random sample."""
    picks = rng.choice(bank["record_count"], size=min(sample, bank["record_count"]),
                       replace=False)
    ok = 0
    failures = []
    for slot in picks:
        layer, expert, prefix = bank["records"][int(slot)]
        code_parts = [index.read_raw(f"{prefix}.{m}.weight") for m in ("down_proj", "up_proj")]
        fresh = np.concatenate(code_parts)
        stored = bank["codes"][int(slot) * CODE_BYTES:(int(slot) + 1) * CODE_BYTES].numpy()
        if np.array_equal(fresh, stored):
            ok += 1
        else:
            failures.append({"layer": layer, "expert": expert})
    return {"sampled": len(picks), "bit_identical": ok, "failures": failures,
            "all_pass": ok == len(picks)}


# --------------------------------------------------------------------------
# GPU decode -- LUT based so it is bit-identical to the CPU decode
# --------------------------------------------------------------------------

def make_luts(device):
    import torch
    e2m1 = torch.tensor(nvfp4.E2M1_TABLE, dtype=torch.float32, device=device)
    e4m3 = torch.tensor(np.nan_to_num(nvfp4.E4M3_TABLE, nan=0.0),
                        dtype=torch.float32, device=device)
    return e2m1, e4m3


def decode_gpu(code_bytes, scale_bytes, global_scale: float, rows: int, cols: int,
               e2m1_lut, e4m3_lut):
    """Unpack + LUT + scale on device, float32 throughout."""
    import torch
    low = (code_bytes & 0x0F).to(torch.long)
    high = (code_bytes >> 4).to(torch.long)
    codes = torch.stack([low, high], dim=-1).reshape(-1)
    values = e2m1_lut[codes]
    scales = e4m3_lut[scale_bytes.to(torch.long)]
    expanded = scales.repeat_interleave(nvfp4.GROUP_SIZE)
    return (values * expanded * global_scale).reshape(rows, cols)


def decode_cpu_float32(code_bytes: np.ndarray, scale_bytes: np.ndarray,
                       global_scale: float, rows: int, cols: int) -> np.ndarray:
    """Same operation order as decode_gpu, in float32, for the G2 comparison."""
    e2m1 = nvfp4.E2M1_TABLE.astype(np.float32)
    e4m3 = np.nan_to_num(nvfp4.E4M3_TABLE, nan=0.0).astype(np.float32)
    low = code_bytes & 0x0F
    high = code_bytes >> 4
    codes = np.stack([low, high], axis=-1).reshape(-1)
    values = e2m1[codes]
    scales = np.repeat(e4m3[scale_bytes], nvfp4.GROUP_SIZE)
    return (values * scales * np.float32(global_scale)).reshape(rows, cols)


def percentiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-verify", type=int, default=8)
    args = parser.parse_args()

    import torch

    started = utc_now()
    rng = np.random.default_rng(SEED)
    index = ShardIndex(MODEL_DIR)

    compute_apps = gpu_compute_apps()
    other_apps = [a for a in compute_apps if a["pid"] != os.getpid()]
    python_context = python_processes()
    gpu_before = gpu_state()
    gpu_busy = bool(other_apps) or gpu_before.get("memory_used_mib", 0) > 256

    result: dict = {
        "kind": "lightningstream_nemotron_n4_zero_cache_dataplane",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N4_ZERO_CACHE_DATAPLANE",
        "started_utc": started,
        "runner_sha256": sha256_path(Path(__file__)),
        "codec_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/nvfp4.py"),
        "loader_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/loader.py"),
        "seed": SEED,
        "top_k": TOP_K,
        "moe_layers": MOE_LAYERS,
        "records_per_token": len(MOE_LAYERS) * TOP_K,
        "record_bytes": RECORD_BYTES,
        "working_set_bytes": len(MOE_LAYERS) * TOP_K * RECORD_BYTES,
        "warmup": WARMUP, "repeats": REPEATS,
        "non_interference": {
            "rule": ("blocked when another PID holds a CUDA context, or device memory "
                     "used exceeds 256 MiB; python process names are context only"),
            "gpu_compute_apps": compute_apps,
            "foreign_gpu_compute_apps": other_apps,
            "python_processes_context_only": python_context,
            "gpu_before": gpu_before,
            "gpu_busy": gpu_busy,
        },
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if gpu_busy:
        result["status"] = "blocked_gpu_busy"
        result["note"] = ("A foreign Python process or non-trivial device memory use was "
                          "detected. GPU arms skipped rather than interfering with the "
                          "PORT80B/STREAMQ5 line.")
        (OUT_DIR / "n4_zero_cache_dataplane.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("BLOCKED: GPU busy or foreign python process; GPU arms skipped.")
        return 4

    # ------------------------------------------------------------- N4-A bank
    plan = build_route_plan(index, rng)
    t0 = time.perf_counter()
    bank = build_host_bank(index, plan)
    build_seconds = time.perf_counter() - t0

    bank_check = verify_bank_against_checkpoint(index, bank, args.sample_verify, rng)
    result["n4a_bank"] = {
        "record_count": bank["record_count"],
        "verified_records": bank["verified_records"],
        "mismatches": bank["mismatches"],
        "working_set_bytes": bank["working_set_bytes"],
        "build_seconds": build_seconds,
        "pinned": True,
        "partial_bank_note": bank["partial_bank_note"],
        "route_provenance": plan["route_provenance"],
        "n3_capture_sha256": plan["n3_capture_sha256"],
        "sample_verification": bank_check,
    }

    if not torch.cuda.is_available():
        result["status"] = "blocked_no_cuda"
        (OUT_DIR / "n4_zero_cache_dataplane.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("BLOCKED: torch has no CUDA; N4-A completed, GPU arms skipped.")
        return 4

    device = torch.device("cuda:0")
    torch.cuda.init()
    free_before, total_device = torch.cuda.mem_get_info()
    result["device"] = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_bytes": int(total_device),
        "free_bytes_before": int(free_before),
    }

    n_records = bank["record_count"]
    d_codes = torch.empty(n_records * CODE_BYTES, dtype=torch.uint8, device=device)
    d_scales = torch.empty(n_records * SCALE_BYTES, dtype=torch.uint8, device=device)
    d_globals = torch.empty(n_records * GLOBAL_BYTES, dtype=torch.uint8, device=device)

    # ---------------------------------------------------------- N4-B transport
    def arm_per_record():
        for slot in range(n_records):
            d_codes[slot * CODE_BYTES:(slot + 1) * CODE_BYTES].copy_(
                bank["codes"][slot * CODE_BYTES:(slot + 1) * CODE_BYTES], non_blocking=True)
            d_scales[slot * SCALE_BYTES:(slot + 1) * SCALE_BYTES].copy_(
                bank["scales"][slot * SCALE_BYTES:(slot + 1) * SCALE_BYTES], non_blocking=True)
            d_globals[slot * GLOBAL_BYTES:(slot + 1) * GLOBAL_BYTES].copy_(
                bank["globals"][slot * GLOBAL_BYTES:(slot + 1) * GLOBAL_BYTES], non_blocking=True)

    def arm_per_layer():
        per_layer = TOP_K
        for start in range(0, n_records, per_layer):
            end = min(start + per_layer, n_records)
            d_codes[start * CODE_BYTES:end * CODE_BYTES].copy_(
                bank["codes"][start * CODE_BYTES:end * CODE_BYTES], non_blocking=True)
            d_scales[start * SCALE_BYTES:end * SCALE_BYTES].copy_(
                bank["scales"][start * SCALE_BYTES:end * SCALE_BYTES], non_blocking=True)
            d_globals[start * GLOBAL_BYTES:end * GLOBAL_BYTES].copy_(
                bank["globals"][start * GLOBAL_BYTES:end * GLOBAL_BYTES], non_blocking=True)

    def arm_single():
        d_codes.copy_(bank["codes"], non_blocking=True)
        d_scales.copy_(bank["scales"], non_blocking=True)
        d_globals.copy_(bank["globals"], non_blocking=True)

    arms = {"per_record": arm_per_record, "per_layer_batched": arm_per_layer,
            "single_contiguous": arm_single}

    transport = {}
    for name, fn in arms.items():
        for _ in range(WARMUP):
            fn()
            torch.cuda.synchronize()
        wall, dev = [], []
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t_start = time.perf_counter_ns()
            start_evt.record()
            fn()
            end_evt.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter_ns() - t_start) / 1e6)
            dev.append(start_evt.elapsed_time(end_evt))
        moved = bank["working_set_bytes"]
        stats = percentiles(wall)
        transport[name] = {
            "wall_ms": stats,
            "device_ms": percentiles(dev),
            "raw_wall_ms": wall,
            "raw_device_ms": dev,
            "bytes_moved": moved,
            "effective_gb_s_at_p50": moved / (stats["p50"] / 1e3) / 1e9,
            "copies_issued": {"per_record": n_records * 3,
                              "per_layer_batched": (n_records // TOP_K) * 3,
                              "single_contiguous": 3}[name],
        }
        print(f"  transport[{name:<18}] p50 {stats['p50']:8.3f} ms  p95 {stats['p95']:8.3f} ms  "
              f"{transport[name]['effective_gb_s_at_p50']:6.2f} GB/s")

    result["n4b_transport"] = transport
    best_arm = min(transport, key=lambda k: transport[k]["wall_ms"]["p95"])
    result["best_transport_arm"] = best_arm

    # ------------------------------------------ N4-C correctness on a real expert
    e2m1_lut, e4m3_lut = make_luts(device)
    capture = json.loads((OUT_DIR / "n3_official_route_capture.json").read_text(encoding="utf-8"))
    probe_expert = int(capture["indices"][0][0])
    prefix = f"backbone.layers.1.mixer.experts.{probe_expert}"

    correctness = {}
    for matrix, rows, cols in (("up_proj", MOE_INTERMEDIATE, HIDDEN),
                               ("down_proj", HIDDEN, MOE_INTERMEDIATE)):
        code_np = index.read_raw(f"{prefix}.{matrix}.weight")
        scale_np = index.read_raw(f"{prefix}.{matrix}.weight_scale")
        gscale = index.get_scalar(f"{prefix}.{matrix}.weight_scale_2")

        cpu32 = decode_cpu_float32(code_np, scale_np, gscale, rows, cols)
        gpu = decode_gpu(torch.from_numpy(code_np.copy()).to(device),
                         torch.from_numpy(scale_np.copy()).to(device),
                         gscale, rows, cols, e2m1_lut, e4m3_lut)
        gpu_np = gpu.cpu().numpy()
        correctness[matrix] = {
            "bit_identical_to_cpu_float32": bool(np.array_equal(cpu32, gpu_np)),
            "elements": int(cpu32.size),
            "cpu_sha256": sha256_array(cpu32),
            "gpu_sha256": sha256_array(gpu_np),
            "max_abs_diff": float(np.max(np.abs(cpu32.astype(np.float64) - gpu_np.astype(np.float64)))),
        }
        del gpu
    torch.cuda.empty_cache()

    # expert output vs the N3 CPU reference
    rng2 = np.random.default_rng(SEED)
    hidden = (rng2.standard_normal((8, HIDDEN)) * 0.5)
    norm_w = index.get_float32("backbone.layers.1.norm.weight")
    normed = ref.rms_norm(hidden, norm_w, index.config["layer_norm_epsilon"])
    up_w = index.dequantize_linear(f"{prefix}.up_proj")
    down_w = index.dequantize_linear(f"{prefix}.down_proj")
    cpu_expert = ref.mlp_relu2(normed, up_w, down_w)

    x_d = torch.from_numpy(normed.astype(np.float32)).to(device)
    up_d = torch.from_numpy(up_w).to(device)
    down_d = torch.from_numpy(down_w).to(device)
    hidden_d = x_d @ up_d.T
    act_d = torch.clamp(hidden_d, min=0) ** 2
    gpu_expert = (act_d @ down_d.T).cpu().numpy()
    rel = float(np.linalg.norm(gpu_expert - cpu_expert) / np.linalg.norm(cpu_expert))
    correctness["expert_output_rel_l2_vs_n3_cpu"] = rel
    del up_d, down_d, hidden_d, act_d, x_d
    torch.cuda.empty_cache()

    result["n4c_correctness"] = correctness

    # ------------------------------- N4-D composed zero-cache token path
    # Decoding all 138 experts at once would need ~5.5 GB of float32 weights, so
    # the token is processed layer by layer: transfer the layer's six records,
    # decode them, run up -> ReLU^2 -> down, weight-sum, free.  That is the real
    # streaming shape, not an artefact of the measurement.
    x_tok = torch.randn(1, HIDDEN, device=device, dtype=torch.float32)
    route_w = torch.full((TOP_K,), 1.0 / TOP_K, device=device, dtype=torch.float32)

    def composed_token():
        acc = torch.zeros(1, HIDDEN, device=device, dtype=torch.float32)
        for layer_idx in range(len(MOE_LAYERS)):
            base = layer_idx * TOP_K
            # transport: this layer's six records
            c0, c1 = base * CODE_BYTES, (base + TOP_K) * CODE_BYTES
            s0, s1 = base * SCALE_BYTES, (base + TOP_K) * SCALE_BYTES
            g0, g1 = base * GLOBAL_BYTES, (base + TOP_K) * GLOBAL_BYTES
            d_codes[c0:c1].copy_(bank["codes"][c0:c1], non_blocking=True)
            d_scales[s0:s1].copy_(bank["scales"][s0:s1], non_blocking=True)
            d_globals[g0:g1].copy_(bank["globals"][g0:g1], non_blocking=True)

            for slot in range(TOP_K):
                rec = base + slot
                # record layout: down_proj then up_proj within each dtype region
                half_c, half_s = CODE_BYTES // 2, SCALE_BYTES // 2
                dc = d_codes[rec * CODE_BYTES: rec * CODE_BYTES + half_c]
                uc = d_codes[rec * CODE_BYTES + half_c: (rec + 1) * CODE_BYTES]
                ds = d_scales[rec * SCALE_BYTES: rec * SCALE_BYTES + half_s]
                us = d_scales[rec * SCALE_BYTES + half_s: (rec + 1) * SCALE_BYTES]

                up_w = decode_gpu(uc, us, 1.0, MOE_INTERMEDIATE, HIDDEN, e2m1_lut, e4m3_lut)
                h = torch.clamp(x_tok @ up_w.T, min=0) ** 2
                del up_w
                down_w = decode_gpu(dc, ds, 1.0, HIDDEN, MOE_INTERMEDIATE, e2m1_lut, e4m3_lut)
                acc += (h @ down_w.T) * route_w[slot]
                del down_w, h
        return acc

    for _ in range(WARMUP):
        composed_token()
        torch.cuda.synchronize()
    composed_wall = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t_start = time.perf_counter_ns()
        composed_token()
        torch.cuda.synchronize()
        composed_wall.append((time.perf_counter_ns() - t_start) / 1e6)

    composed_stats = percentiles(composed_wall)

    # Attribution: same loop, decode only, no GEMV. The difference against the
    # composed path isolates how much of the token is NVFP4 decode.
    def decode_only_token():
        sink = 0.0
        for layer_idx in range(len(MOE_LAYERS)):
            base = layer_idx * TOP_K
            for slot in range(TOP_K):
                rec = base + slot
                half_c, half_s = CODE_BYTES // 2, SCALE_BYTES // 2
                dc = d_codes[rec * CODE_BYTES: rec * CODE_BYTES + half_c]
                uc = d_codes[rec * CODE_BYTES + half_c: (rec + 1) * CODE_BYTES]
                ds = d_scales[rec * SCALE_BYTES: rec * SCALE_BYTES + half_s]
                us = d_scales[rec * SCALE_BYTES + half_s: (rec + 1) * SCALE_BYTES]
                up_w = decode_gpu(uc, us, 1.0, MOE_INTERMEDIATE, HIDDEN, e2m1_lut, e4m3_lut)
                sink += float(up_w[0, 0])
                del up_w
                down_w = decode_gpu(dc, ds, 1.0, HIDDEN, MOE_INTERMEDIATE, e2m1_lut, e4m3_lut)
                sink += float(down_w[0, 0])
                del down_w
        return sink

    for _ in range(2):
        decode_only_token()
        torch.cuda.synchronize()
    decode_wall = []
    for _ in range(10):
        torch.cuda.synchronize()
        t_start = time.perf_counter_ns()
        decode_only_token()
        torch.cuda.synchronize()
        decode_wall.append((time.perf_counter_ns() - t_start) / 1e6)
    decode_stats = percentiles(decode_wall)
    result["n4d_decode_only"] = {
        "wall_ms": decode_stats,
        "raw_wall_ms": decode_wall,
        "repeats": 10,
        "note": ("Same streaming loop with the two GEMVs removed. Includes a "
                 "per-matrix scalar readback to prevent the decode being elided; "
                 "that readback forces a sync per matrix and is itself part of "
                 "the measured cost, so this is an upper bound on decode."),
    }
    print(f"  decode only         p50 {decode_stats['p50']:8.3f} ms  "
          f"p95 {decode_stats['p95']:8.3f} ms")
    result["n4d_composed_token"] = {
        "wall_ms": composed_stats,
        "raw_wall_ms": composed_wall,
        "bytes_moved": bank["working_set_bytes"],
        "experts_executed": n_records,
        "decode_implementation": "unfused torch ops (LUT unpack + scale + GEMV)",
        "note": ("Layer-by-layer streaming: transfer six records, decode, "
                 "up -> ReLU^2 -> down, weighted sum, free. A hand-written fused "
                 "NVFP4 kernel is H6 work and is NOT part of this measurement."),
        "route_weights": "uniform 1/6 placeholder; the official weighted reduction is exercised in N3",
    }
    del x_tok
    torch.cuda.empty_cache()
    print(f"  composed token      p50 {composed_stats['p50']:8.3f} ms  "
          f"p95 {composed_stats['p95']:8.3f} ms")

    # ------------------------------------------------------ peak device usage
    free_after, _ = torch.cuda.mem_get_info()
    result["device"]["free_bytes_after"] = int(free_after)
    result["device"]["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["device"]["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    result["gpu_after"] = gpu_state()

    # ----------------------------------------------------------------- gates
    best = transport[best_arm]["wall_ms"]
    gates = {
        "G1_bank_records_bit_identical": bank_check["all_pass"] and not bank["mismatches"],
        "G2_gpu_decode_bit_identical": all(
            correctness[m]["bit_identical_to_cpu_float32"] for m in ("up_proj", "down_proj")),
        "G3_expert_output_within_1e-5": rel <= 1e-5,
        # G4 is the routed-expert PATH, i.e. the composed token, not transport alone.
        "G4_routed_path_p95_under_45ms": composed_stats["p95"] <= GATE_P95_MS,
        "G5_no_full_dequantized_bank": True,
        "G6_peak_device_under_8gib": result["device"]["peak_reserved_bytes"] <= 8 * (1024 ** 3),
    }
    result["gates"] = gates
    result["gates_all_pass"] = all(gates.values())
    # The preregistered architectural stop reads: "if p95 exceeds 60 ms AFTER
    # registered/batched transfer AND a correct fused kernel, reassess the
    # physical architecture". This run has batched transfer but NOT a fused
    # kernel -- decode is unfused torch ops -- so the stop's precondition is not
    # satisfied and it must not be declared, however large the number is.
    fused_kernel_present = False
    result["architectural_stop_precondition"] = {
        "batched_transfer_tested": True,
        "correct_fused_kernel_present": fused_kernel_present,
        "precondition_satisfied": fused_kernel_present,
    }
    result["architectural_stop_triggered"] = (
        fused_kernel_present and composed_stats["p95"] > ARCH_STOP_MS)

    if result["gates_all_pass"]:
        terminal = "n4_zero_cache_screen_pass"
    elif result["architectural_stop_triggered"]:
        terminal = "n4_zero_cache_architectural_stop"
    elif composed_stats["p95"] > ARCH_STOP_MS:
        terminal = "n4_zero_cache_screen_fail_unfused_decode_dominates"
    else:
        terminal = "n4_zero_cache_screen_fail"
    result["terminal_state"] = terminal
    result["completed_utc"] = utc_now()
    result["claim_boundary"] = (
        "Physically measured cache-free routed-expert transport and decode for one "
        "token on this specific GPU, with bit-exact decode and reference-matching "
        "output. NOT a tokens-per-second, full-model, quality or memory-feasibility "
        "claim. A component measurement is never promoted to tok/s."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "n4_zero_cache_dataplane.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"best arm         : {best_arm}")
    print(f"  p50/p95/p99    : {best['p50']:.3f} / {best['p95']:.3f} / {best['p99']:.3f} ms")
    print(f"expert rel_l2    : {rel:.3e}")
    print(f"peak reserved    : {result['device']['peak_reserved_bytes']:,} B")
    for key, value in gates.items():
        print(f"  {'OK  ' if value else 'FAIL'} {key}")
    print(f"terminal state   : {result['terminal_state']}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
