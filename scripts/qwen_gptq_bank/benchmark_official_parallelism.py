from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from moe_lab.qwen_gptq_bank import codes_from_quantized, official_pure_gptq_projection
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
GSQ = ROOT / "third_party/GSQ"
CALIBRATION = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration/layer_00.safetensors"
LOCK = ROOT / "reports/qwen_gptq_bank/P0_OFFICIAL_PARALLEL_FALLBACK_LOCK.md"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_official_parallelism_benchmark.json"


def digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def task(expert: int):
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    with safe_open(CALIBRATION, framework="pt", device="cpu") as handle:
        x = handle.get_tensor("moe_input")[expert].to(device)
    base = f"model.layers.0.mlp.experts.{expert}"
    names = {kind: f"{base}.{kind}_proj.weight" for kind in ("gate", "up", "down")}
    loaded = load_checkpoint_tensors(MODEL, list(names.values()), checkpoint_weight_map(MODEL))
    weights = {kind: loaded[name].to(device).contiguous() for kind, name in names.items()}
    z = F.silu(F.linear(x, weights["gate"])) * F.linear(x, weights["up"])
    started = time.perf_counter()
    items = {
        "gate": official_pure_gptq_projection(weights["gate"], x, GSQ),
        "up": official_pure_gptq_projection(weights["up"], x, GSQ),
        "down": official_pure_gptq_projection(weights["down"], z, GSQ),
    }
    torch.cuda.synchronize(device)
    hashes = {}
    for kind, item in items.items():
        code = codes_from_quantized(item.weight.unsqueeze(0), item.scales.to(torch.bfloat16).unsqueeze(0))[0]
        hashes[kind] = {
            "codes": digest_tensor(code),
            "scales": digest_tensor(item.scales.to(torch.bfloat16)),
        }
    return {
        "expert": expert, "seconds": time.perf_counter() - started,
        "hashes": hashes, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite parallelism benchmark")
    ctx = mp.get_context("spawn")
    modes = []
    control_hash = None
    for workers in (1, 2, 4, 8):
        experts = list(range(workers))
        started = time.perf_counter()
        with ctx.Pool(workers) as pool:
            results = pool.map(task, experts)
        wall = time.perf_counter() - started
        current_control = results[0]["hashes"]
        if control_hash is None:
            control_hash = current_control
        exact = current_control == control_hash
        modes.append({
            "workers": workers, "experts": workers, "wall_seconds": wall,
            "experts_per_second": workers / wall, "control_hash_exact": exact,
            "maximum_worker_peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in results),
            "results": results,
        })
        print(json.dumps({key: modes[-1][key] for key in (
            "workers", "wall_seconds", "experts_per_second", "control_hash_exact",
            "maximum_worker_peak_cuda_allocated_bytes",
        )}), flush=True)
    eligible = [row for row in modes if row["control_hash_exact"]]
    selected = max(eligible, key=lambda row: row["experts_per_second"])
    payload = {
        "kind": "qwen_gptq_bank_p0_official_parallelism_benchmark",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass", "fallback_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "modes": modes, "selected_workers": selected["workers"],
        "selected_experts_per_second": selected["experts_per_second"],
        "projected_full_bank_hours": 6_144 / selected["experts_per_second"] / 3_600,
        "claim_boundary": "Throughput/control benchmark only; no bank expert was produced.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "selected_workers": payload["selected_workers"],
        "projected_full_bank_hours": payload["projected_full_bank_hours"],
    }, indent=2))
