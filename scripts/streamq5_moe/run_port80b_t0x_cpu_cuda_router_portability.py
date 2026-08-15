from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUN_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0x_router_portability"
PREREG = REPORTS / "PORT80B_T0X_CPU_CUDA_ROUTER_PORTABILITY_PREREGISTRATION_2026-08-13.md"
LOCK = REPORTS / "port80b_t0x_router_portability_lock.json"
VERIFIER = ROOT / "scripts" / "streamq5_moe" / "verify_port80b_t0x_cpu_cuda_router_portability.py"
R6_RAW = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0r6d_router_diagnostic" / "t0r6d_router_raw.safetensors"
R6_JSON = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0r6d_router_diagnostic" / "t0r6d_router_diagnostic.json"
SHARD = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-Coder-Next" / "snapshots" / "a19358a7659bd1f564300250ee189120c49a562f" / "model-00001-of-00040.safetensors"

EXPECTED = {
    "r6_raw_sha256": "42b9eb25748ce0722f7b3f7c5612069081314eae51b8741a18c39b17abcbdb72",
    "r6_json_sha256": "fd35d86e0bc8679d614ac209dc3d3a679ec735307db7455e670e37947780f797",
    "shard_sha256": "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def route(hidden: torch.Tensor, weight: torch.Tensor):
    logits = F.linear(hidden.reshape(-1, 2048), weight)
    probs = torch.softmax(logits, dtype=torch.float32, dim=-1)
    values, ids = torch.topk(probs, 10, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True)
    return logits, probs, ids, values.to(logits.dtype)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint16), b.view(torch.uint16))


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source_checks = {
        "runner": sha256(Path(__file__)),
        "verifier": sha256(VERIFIER),
        "prereg": sha256(PREREG),
    }
    if source_checks != lock["source_sha256"]:
        raise RuntimeError("source lock mismatch before CUDA initialization")
    input_checks = {"r6_raw_sha256": sha256(R6_RAW), "r6_json_sha256": sha256(R6_JSON), "shard_sha256": sha256(SHARD)}
    if input_checks != EXPECTED:
        raise RuntimeError("input hash mismatch before CUDA initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 1 << 30:
        raise RuntimeError("less than 1 GiB free VRAM")

    with safe_open(str(R6_RAW), framework="pt", device="cpu") as f:
        hidden = f.get_tensor("official_gate_input").contiguous()
        archived_logits = f.get_tensor("official_logits").contiguous()
        archived_ids = f.get_tensor("official_ids").contiguous()
        archived_weights = f.get_tensor("official_weights").contiguous()
    with safe_open(str(SHARD), framework="pt", device="cpu") as f:
        gate_weight = f.get_tensor("model.layers.0.mlp.gate.weight").contiguous()
    if hidden.shape != (1, 16, 2048) or hidden.dtype != torch.bfloat16:
        raise RuntimeError("hidden contract mismatch")
    if gate_weight.shape != (512, 2048) or gate_weight.dtype != torch.bfloat16:
        raise RuntimeError("weight contract mismatch")

    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    cpu = tuple(x.detach().cpu().contiguous() for x in route(hidden, gate_weight))
    cpu_replay = {
        "logits": bits_equal(cpu[0], archived_logits),
        "ids": torch.equal(cpu[2], archived_ids),
        "weights": bits_equal(cpu[3], archived_weights),
    }
    if not all(cpu_replay.values()):
        raise RuntimeError(f"CPU archive replay mismatch: {cpu_replay}")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    h_gpu = hidden.to(device)
    w_gpu = gate_weight.to(device)
    gpu_calls = []
    for _ in range(2):
        out = route(h_gpu, w_gpu)
        torch.cuda.synchronize(device)
        gpu_calls.append(tuple(x.detach().cpu().contiguous() for x in out))
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak >= 256 << 20:
        raise RuntimeError("CUDA allocation exceeded 256 MiB")

    cpu_ids, gpu_ids = cpu[2], gpu_calls[0][2]
    rows = []
    r6 = json.loads(R6_JSON.read_text(encoding="utf-8"))
    for i in range(16):
        cpu_list = [int(x) for x in cpu_ids[i].tolist()]
        gpu_list = [int(x) for x in gpu_ids[i].tolist()]
        rows.append({
            "row": i,
            "ordered_ids_equal": cpu_list == gpu_list,
            "id_set_equal": set(cpu_list) == set(gpu_list),
            "cpu_ids": cpu_list,
            "gpu_ids": gpu_list,
            "symmetric_difference": sorted(set(cpu_list) ^ set(gpu_list)),
            "cpu_probability_margin": float(r6["rows"][i]["probability_margin"]),
            "cpu_boundary_tie_expert_ids": r6["rows"][i]["boundary_tie_expert_ids"],
            "selected_weights_bit_equal": bits_equal(cpu[3][i], gpu_calls[0][3][i]),
            "logit_max_abs": float((cpu[0][i].float() - gpu_calls[0][0][i].float()).abs().max()),
        })

    cuda_repeat = {
        "logits": bits_equal(gpu_calls[0][0], gpu_calls[1][0]),
        "probs": torch.equal(gpu_calls[0][1], gpu_calls[1][1]),
        "ids": torch.equal(gpu_calls[0][2], gpu_calls[1][2]),
        "weights": bits_equal(gpu_calls[0][3], gpu_calls[1][3]),
    }
    all_finite = all(torch.isfinite(x.float()).all().item() for call in (cpu, gpu_calls[0], gpu_calls[1]) for x in (call[0], call[1], call[3]))
    exact = all(r["ordered_ids_equal"] and r["selected_weights_bit_equal"] for r in rows) and all(cuda_repeat.values()) and all_finite
    verdict = "exact_cross_backend_pass" if exact else "cross_backend_negative"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RUN_DIR / "t0x_cpu_cuda_router_raw.safetensors"
    result_path = RUN_DIR / "t0x_cpu_cuda_router_result.json"
    save_file({
        "hidden": hidden,
        "gate_weight": gate_weight,
        "cpu_logits": cpu[0], "cpu_probs": cpu[1], "cpu_ids": cpu[2], "cpu_weights": cpu[3],
        "gpu1_logits": gpu_calls[0][0], "gpu1_probs": gpu_calls[0][1], "gpu1_ids": gpu_calls[0][2], "gpu1_weights": gpu_calls[0][3],
        "gpu2_logits": gpu_calls[1][0], "gpu2_probs": gpu_calls[1][1], "gpu2_ids": gpu_calls[1][2], "gpu2_weights": gpu_calls[1][3],
    }, str(raw_path))
    props = torch.cuda.get_device_properties(device)
    result = {
        "kind": "port80b_t0x_cpu_cuda_router_portability",
        "status": verdict,
        "overall_pass": exact,
        "verdict": verdict,
        "rows": rows,
        "cpu_archive_replay": cpu_replay,
        "cuda_repeat": cuda_repeat,
        "all_finite": all_finite,
        "ordered_id_equal_rows": sum(r["ordered_ids_equal"] for r in rows),
        "id_set_equal_rows": sum(r["id_set_equal"] for r in rows),
        "weight_bit_equal_rows": sum(r["selected_weights_bit_equal"] for r in rows),
        "zero_cpu_margin_rows": sum(r["cpu_probability_margin"] == 0.0 for r in rows),
        "raw_artifact": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha256(raw_path),
        "input_sha256": input_checks,
        "source_sha256": source_checks,
        "resources": {"free_vram_before": int(free_bytes), "peak_cuda_allocated": peak},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": props.name, "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor]},
        "bank_built": False,
        "host_registered": False,
        "claim_boundary": "Diagnostic CPU/CUDA router portability for 16 archived layer-0 rows only.",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": verdict, "ordered_id_equal_rows": result["ordered_id_equal_rows"], "weight_bit_equal_rows": result["weight_bit_equal_rows"], "cuda_repeat": cuda_repeat, "peak_cuda_allocated": peak}, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
