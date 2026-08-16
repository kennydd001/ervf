"""Shared helpers for the post-V6 exact campaign.

CuPy and the model runtime are imported lazily so install/verification can run
without creating a CUDA context.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
PRO = HERE.parent
REPO = PRO.parent
RESULTS = PRO / "results" / "pro_max_v2"
HISTORY = RESULTS / "history"
LOGS = RESULTS / "logs"

for p in (str(PRO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs() -> None:
    for p in (RESULTS, HISTORY, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def result_path(name: str) -> Path:
    ensure_dirs()
    return RESULTS / name


def archive_existing(path: Path) -> Path | None:
    ensure_dirs()
    if not path.exists():
        return None
    target = HISTORY / f"{path.stem}__{slug()}{path.suffix}"
    i = 1
    while target.exists():
        target = HISTORY / f"{path.stem}__{slug()}_{i}{path.suffix}"
        i += 1
    shutil.move(str(path), str(target))
    return target


def write_json(path: Path, payload: Any, *, archive: bool = True) -> None:
    ensure_dirs()
    if archive:
        archive_existing(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def write_text(path: Path, text: str, *, archive: bool = True) -> None:
    ensure_dirs()
    if archive:
        archive_existing(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_text(cmd: list[str], timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           check=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return (p.stdout or p.stderr or "").strip()


def git_head() -> str | None:
    x = run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    if not x or x.startswith("ERROR"):
        return None
    return x.splitlines()[0]


def git_status() -> list[str]:
    x = run_text(["git", "-C", str(REPO), "status", "--short"])
    return [] if not x or x.startswith("ERROR") else x.splitlines()


def nvidia_snapshot() -> str:
    return run_text([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,"
        "power.draw,clocks.sm,clocks.mem,pstate",
        "--format=csv,noheader",
    ])


def gpu_processes() -> list[str]:
    x = run_text([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if not x or x.startswith("ERROR"):
        return []
    return [line for line in x.splitlines() if line.strip()]


def require_gpu_free() -> None:
    procs = gpu_processes()
    if procs:
        raise RuntimeError("Another CUDA process is active:\n  " + "\n  ".join(procs))


def environment(extra: tuple[Path, ...] = ()) -> dict[str, Any]:
    import hashlib
    hashes = {}
    for p in extra:
        try:
            hashes[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            hashes[str(p)] = None
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "git_head": git_head(),
        "git_status": git_status(),
        "nvidia_smi": nvidia_snapshot(),
        "model_dir": str(REPO / "models" / "nemotron_3_5_lightning_v35"),
        "source_hashes": hashes,
    }


def percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "max": None}
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size), "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def pointer(obj: Any) -> int:
    return int(obj.data.ptr)


def same_bits(a: Any, b: Any) -> bool:
    import cupy as cp
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(cp.all(a.view(cp.uint8) == b.view(cp.uint8)).item())


def cuda_time(fn: Callable[[], None], warmup: int = 10, repeats: int = 100) -> list[float]:
    import cupy as cp
    for _ in range(warmup):
        fn()
    cp.cuda.Device(0).synchronize()
    out: list[float] = []
    for _ in range(repeats):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record(); fn(); b.record(); b.synchronize()
        out.append(float(cp.cuda.get_elapsed_time(a, b)))
    return out


@dataclass
class V6Bundle:
    rt: Any
    dense: Any
    down: Any
    up: Any
    restore_selective: Callable[[], None]
    restore_moe: Callable[[], None]
    counters: dict[str, int]
    capacity: int

    def close(self) -> None:
        try:
            self.restore_selective()
        except Exception:
            pass
        try:
            self.restore_moe()
        except Exception:
            pass
        self.rt = None
        self.dense = None
        self.down = None
        self.up = None
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def new_v6_bundle(capacity: int = 72) -> V6Bundle:
    """Build the exact V6 stack but do not capture a graph yet."""
    from graph_e1f22 import _new_runtime
    from ervf_dense import DenseERVF
    from down_proj_batch_kernels import DownProjBatchKernels
    from up_proj_batch_kernels import UpProjBatchKernels
    from selective_ervf_v3 import _install_selective
    from moe_dev_batched import install_batched_moe_dev
    from layer_capacity import apply_nonuniform_capacity

    rt = _new_runtime(capacity)
    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()
    rt.enable_cache(capacity)
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    restore_selective, counters = _install_selective(rt, dense)
    restore_moe = install_batched_moe_dev(rt, down, up)
    return V6Bundle(rt, dense, down, up, restore_selective, restore_moe,
                    counters, capacity)


def drop_graph_state(rt: Any) -> None:
    """Discard one captured graph before recapture without retaining its buffers."""
    rt._graph = None
    rt.graph_mode = False
    names = (
        "_tok_dev", "_pos_dev", "_am_max", "_am_idx", "_embed_pinned",
        "_stage_mem", "_stage_np", "_ring_mem", "_ring_np", "_graph_stream",
    )
    for name in names:
        if hasattr(rt, name):
            try:
                delattr(rt, name)
            except Exception:
                pass
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def capture_v6(bundle: V6Bundle) -> int:
    from layer_capacity import apply_nonuniform_capacity
    import cupy as cp
    rt = bundle.rt
    drop_graph_state(rt)
    rt.enable_cache(bundle.capacity)
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    free0 = int(cp.cuda.Device(0).mem_info[0])
    rt.setup_graph()
    free1 = int(cp.cuda.Device(0).mem_info[0])
    return int(getattr(rt, "graph_extra_vram_bytes", free0 - free1))


def prompt_set(mode: str):
    from graph_e1f22 import _load_prompt_set
    return _load_prompt_set(mode)


def run_graph(rt: Any, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    """Safe prompt staging followed by timed autoregressive graph replays."""
    import time
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
        rt._graph_stream.synchronize()
    first_slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    cur = int(rt.ring_harvest(first_slot, 1)[0])
    ids = [cur]
    times: list[float] = []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        cur = int(rt.ring_harvest(slot, 1)[0])
        times.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(cur)
    return ids, times


def run_arm(rt: Any, prompts: list[dict[str, Any]], n: int) -> dict[str, Any]:
    ids: dict[str, list[int]] = {}
    samples: list[float] = []
    for p in prompts:
        out, ms = run_graph(rt, [int(x) for x in p["prompt_ids"]], n)
        ids[p["prompt"]] = out
        samples.extend(ms)
    return {"ids": ids, "timing_ms": percentiles(samples), "raw_timing_ms": samples}


def graph_dot(rt: Any) -> str:
    try:
        text = rt._graph.debug_dot_str()
        return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
    except Exception:
        return ""


def compare_arms(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for prompt, aa in a["ids"].items():
        bb = b["ids"].get(prompt, [])
        out[prompt] = {"identical": aa == bb,
                       "first_divergence": first_divergence(aa, bb)}
    return out


def status_from_gates(gates: dict[str, Any], required: tuple[str, ...]) -> str:
    return "pass" if all(bool(gates.get(k)) for k in required) else "gate_failed"
