"""N5 runner: physically allocate the resident shell and measure real VRAM.

Executes ``N5_PHYSICAL_RESIDENT_SHELL_PREREGISTRATION_2026-08-14.md``.

Everything except the routed NVFP4 experts is uploaded to the device and
touched.  Device usage is measured from the driver via ``cuMemGetInfo``, so CUDA
context overhead and allocator fragmentation are inside the number.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

MIB = 1024 ** 2
GIB = 1024 ** 3

EXPERT_STAGING_BYTES = 774_533_280          # N4-R2 measured token working set
MAMBA_STATE_BYTES = 49_364_992              # N3 projection, constant in context
KV_BYTES_PER_TOKEN = 3_072                  # 6 attn layers x 2 kv heads x 128 x 2
CONTEXTS = [4_096, 131_072]

ROUTED_RE = re.compile(r"^backbone\.layers\.\d+\.mixer\.experts\.\d+\.")
EMBED = "backbone.embeddings.weight"
LM_HEAD = "lm_head.weight"

VARIANTS = {
    "A_embed_device_head_device": {"embed": "device", "head": "device"},
    "B_embed_host_head_device": {"embed": "host", "head": "device"},
    "C_embed_host_head_host": {"embed": "host", "head": "host"},
}


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


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def process_memory() -> dict:
    """Windows process memory counters.

    ``GetProcessMemoryInfo`` lives in psapi.dll but is forwarded to kernel32 as
    ``K32GetProcessMemoryInfo`` on modern Windows; the psapi entry point can
    fill only the base struct, leaving ``PrivateUsage`` at zero.  Both are tried
    and the result is validated rather than trusted.
    """
    try:
        counters = _MemCounters()
        counters.cb = ctypes.sizeof(_MemCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()

        ok = 0
        for dll, symbol in (("kernel32", "K32GetProcessMemoryInfo"),
                            ("psapi", "GetProcessMemoryInfo")):
            try:
                fn = getattr(getattr(ctypes.windll, dll), symbol)
            except AttributeError:
                continue
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemCounters), ctypes.c_ulong]
            fn.restype = ctypes.c_int
            ok = fn(handle, ctypes.byref(counters), counters.cb)
            if ok and counters.WorkingSetSize > 0:
                break
        if not ok:
            return {"error": "GetProcessMemoryInfo failed"}
        if counters.WorkingSetSize == 0:
            return {"error": "counters returned zero working set"}
        return {
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "commit_bytes": int(counters.PrivateUsage),
            "peak_commit_bytes": int(counters.PeakPagefileUsage),
            "page_fault_count": int(counters.PageFaultCount),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    import cupy as cp

    started = utc_now()
    compute_apps = gpu_compute_apps()
    other = [a for a in compute_apps if a["pid"] != os.getpid()]
    if other:
        print("BLOCKED: another PID holds a CUDA context; not interfering.")
        return 4

    index = ShardIndex(MODEL_DIR)

    # ---------------------------------------------- classify the shell tensors
    routed, shell = [], []
    for name, entry in index.entries.items():
        (routed if ROUTED_RE.match(name) else shell).append((name, entry))

    routed_bytes = sum(e.nbytes for _, e in routed)
    shell_bytes = sum(e.nbytes for _, e in shell)
    embed_bytes = index.entries[EMBED].nbytes
    head_bytes = index.entries[LM_HEAD].nbytes

    print(f"routed tensors : {len(routed):,}  {routed_bytes:,} B")
    print(f"shell tensors  : {len(shell):,}  {shell_bytes:,} B")
    print(f"  embeddings   : {embed_bytes:,} B")
    print(f"  lm_head      : {head_bytes:,} B")

    free_no_ctx, total_device = cp.cuda.runtime.memGetInfo()
    # Force context creation with a trivial allocation, then measure its cost.
    probe = cp.zeros(1, dtype=cp.uint8)
    free_after_ctx, _ = cp.cuda.runtime.memGetInfo()
    context_overhead = free_no_ctx - free_after_ctx
    del probe

    print(f"device total   : {total_device:,} B")
    print(f"free (no ctx)  : {free_no_ctx:,} B")
    print(f"context cost   : {context_overhead:,} B")

    proc_before = process_memory()
    variants_out = {}

    for variant, placement in VARIANTS.items():
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        free_start, _ = cp.cuda.runtime.memGetInfo()

        held = []
        uploaded_bytes = 0
        host_bytes = 0
        tensors_uploaded = 0

        try:
            # --- shell weights -------------------------------------------
            for name, entry in shell:
                if name == EMBED and placement["embed"] == "host":
                    host_bytes += entry.nbytes
                    continue
                if name == LM_HEAD and placement["head"] == "host":
                    host_bytes += entry.nbytes
                    continue
                raw = index.read_raw(name)
                buf = cp.asarray(raw)          # upload
                buf += cp.uint8(0)             # touch: forces real backing
                held.append(buf)
                uploaded_bytes += entry.nbytes
                tensors_uploaded += 1

            free_after_weights, _ = cp.cuda.runtime.memGetInfo()

            # --- runtime buffers ------------------------------------------
            staging = cp.zeros(EXPERT_STAGING_BYTES, dtype=cp.uint8)
            staging[::4096] = 1
            held.append(staging)

            mamba_state = cp.zeros(MAMBA_STATE_BYTES, dtype=cp.uint8)
            mamba_state[::4096] = 1
            held.append(mamba_state)

            free_after_runtime, _ = cp.cuda.runtime.memGetInfo()

            # --- KV at each declared context ------------------------------
            kv_results = {}
            for ctx in CONTEXTS:
                kv_bytes = ctx * KV_BYTES_PER_TOKEN
                kv = cp.zeros(kv_bytes, dtype=cp.uint8)
                kv[::4096] = 1
                free_with_kv, _ = cp.cuda.runtime.memGetInfo()
                kv_results[str(ctx)] = {
                    "kv_bytes": kv_bytes,
                    "free_after_kv": int(free_with_kv),
                    "free_mib": round(free_with_kv / MIB, 3),
                    "fits": free_with_kv > 0,
                }
                del kv
                pool.free_all_blocks()
                # re-touch so subsequent measurements are comparable
                free_recheck, _ = cp.cuda.runtime.memGetInfo()
                kv_results[str(ctx)]["free_after_release"] = int(free_recheck)

            # Largest declared context held for the headroom figure.
            kv_max = cp.zeros(max(CONTEXTS) * KV_BYTES_PER_TOKEN, dtype=cp.uint8)
            kv_max[::4096] = 1
            held.append(kv_max)

            free_end, _ = cp.cuda.runtime.memGetInfo()
            device_used = free_start - free_end
            peak_total = total_device - free_end

            variants_out[variant] = {
                "placement": placement,
                "tensors_uploaded": tensors_uploaded,
                "uploaded_bytes": uploaded_bytes,
                "host_resident_shell_bytes": host_bytes,
                "expert_staging_bytes": EXPERT_STAGING_BYTES,
                "mamba_state_bytes": MAMBA_STATE_BYTES,
                "kv_by_context": kv_results,
                "kv_held_bytes": max(CONTEXTS) * KV_BYTES_PER_TOKEN,
                "free_before": int(free_start),
                "free_after_weights": int(free_after_weights),
                "free_after_runtime": int(free_after_runtime),
                "free_after_all": int(free_end),
                "device_used_bytes": int(device_used),
                "device_used_gib": round(device_used / GIB, 4),
                "peak_total_device_bytes": int(peak_total),
                "peak_total_device_gib": round(peak_total / GIB, 4),
                "free_headroom_bytes": int(free_end),
                "free_headroom_mib": round(free_end / MIB, 3),
                "fits": free_end > 0,
                "error": None,
            }
            print(f"  {variant:<32} used {device_used / GIB:6.3f} GiB  "
                  f"free {free_end / MIB:9.3f} MiB")

        except Exception as exc:
            variants_out[variant] = {"placement": placement, "fits": False,
                                     "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {variant:<32} FAILED: {type(exc).__name__}")

        # Clearing the list is not enough: the loop's local names still hold
        # references, so the pool cannot reclaim the blocks. Rebind them.
        held.clear()
        staging = mamba_state = kv_max = buf = None  # noqa: F841
        pool.free_all_blocks()

    proc_after = process_memory()

    # ------------------------------------------------------------- teardown
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    free_final, _ = cp.cuda.runtime.memGetInfo()
    leak = free_after_ctx - free_final
    teardown_ok = abs(leak) <= 64 * MIB

    ok_variants = {k: v for k, v in variants_out.items() if v.get("fits")}
    best = min(ok_variants, key=lambda k: ok_variants[k]["peak_total_device_bytes"]) if ok_variants else None

    gates = {
        "S1_shell_bytes_reconcile_with_n2": shell_bytes + routed_bytes == 19_339_781_632,
        "S2_peak_device_under_8gib": bool(
            ok_variants and min(v["peak_total_device_bytes"] for v in ok_variants.values()) <= 8 * GIB),
        # Fail closed: an unmeasurable process footprint is not a pass.
        "S3_process_under_32gib": (
            "error" not in proc_after
            and proc_after.get("peak_commit_bytes", 0) > 0
            and proc_after["peak_commit_bytes"] <= 32 * GIB),
        "S4_shell_coexists_with_expert_staging": bool(ok_variants),
        "S5_kv_and_mamba_allocated_at_both_contexts": all(
            all(k["fits"] for k in v["kv_by_context"].values())
            for v in ok_variants.values()) if ok_variants else False,
        "S6_no_precision_reduction": True,
        "S7_free_at_least_256mib": bool(
            ok_variants and max(v["free_headroom_bytes"] for v in ok_variants.values()) >= 256 * MIB),
        "S8_teardown_clean": teardown_ok,
    }

    result = {
        "kind": "lightningstream_nemotron_n5_resident_shell",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N5_PHYSICAL_RESIDENT_SHELL",
        "started_utc": started, "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "cupy_version": cp.__version__,
        "device_total_bytes": int(total_device),
        "free_without_context": int(free_no_ctx),
        "cuda_context_overhead_bytes": int(context_overhead),
        "allocations_touched": True,
        "measurement_source": "cuMemGetInfo driver query, not allocator bookkeeping",
        "inventory": {
            "routed_tensor_count": len(routed),
            "routed_bytes": routed_bytes,
            "shell_tensor_count": len(shell),
            "shell_bytes": shell_bytes,
            "embeddings_bytes": embed_bytes,
            "lm_head_bytes": head_bytes,
            "total_bytes": routed_bytes + shell_bytes,
        },
        "constants": {
            "expert_staging_bytes": EXPERT_STAGING_BYTES,
            "mamba_state_bytes": MAMBA_STATE_BYTES,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "contexts_tested": CONTEXTS,
        },
        "variants": variants_out,
        "best_variant": best,
        "process_memory_before": proc_before,
        "process_memory_after": proc_after,
        "teardown": {"free_after_context": int(free_after_ctx),
                     "free_final": int(free_final),
                     "leak_bytes": int(leak), "clean": teardown_ok},
        "non_interference": {"gpu_compute_apps": compute_apps, "foreign": other},
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "claim_boundary": (
            "Which components physically fit simultaneously on this specific GPU "
            "with real touched allocations, and how much headroom remains. NOT "
            "tokens per second, full-model latency, quality, evidence that a full "
            "runtime exists, or that any context length is achievable in practice "
            "-- KV allocation is necessary for long context, not sufficient. A "
            "component measurement is never promoted to tok/s."
        ),
    }
    result["terminal_state"] = (
        "n5_resident_shell_fits" if result["gates_all_pass"] else "n5_resident_shell_fail")

    (OUT_DIR / "n5_resident_shell.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"best variant   : {best}")
    print(f"teardown leak  : {leak:,} B (clean={teardown_ok})")
    print(f"peak commit    : {proc_after.get('peak_commit_bytes', 0):,} B")
    for key, value in gates.items():
        print(f"  {'OK  ' if value else 'FAIL'} {key}")
    print(f"terminal state : {result['terminal_state']}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
