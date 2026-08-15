from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open

from moe_lab.qwen_gptq_bank import (
    codes_from_quantized, fastgrid_pure_gptq_projection, official_pure_gptq_projection,
)
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
GSQ_ROOT = ROOT / "third_party/GSQ"
LOCK = ROOT / "reports/qwen_gptq_bank/p0_batched_equivalence_lock.json"
CALIBRATION_RESULT = ROOT / "reports/qwen_gptq_bank/p0_calibration_capture_result.json"
CALIBRATION_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration"
ATTEMPT = ROOT / "reports/qwen_gptq_bank/P0_FASTGRID_EQUIVALENCE_ATTEMPT_H.md"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_fastgrid_equivalence_result_h.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_FASTGRID_EQUIVALENCE_REPORT_H.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weights(layer: int, experts: list[int], device, weight_map):
    names = []
    mapping = {}
    for expert in experts:
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        mapping[expert] = {
            kind: f"{base}.{kind}_proj.weight" for kind in ("gate", "up", "down")
        }
        names.extend(mapping[expert].values())
    tensors = load_checkpoint_tensors(MODEL, names, weight_map)
    return {
        kind: torch.stack([tensors[mapping[expert][kind]] for expert in experts]).to(device).contiguous()
        for kind in ("gate", "up", "down")
    }


def original_down_inputs(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    rows = []
    for index in range(x.shape[0]):
        rows.append(F.silu(F.linear(x[index], gate[index])) * F.linear(x[index], up[index]))
    return torch.stack(rows).contiguous()


def codes(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return codes_from_quantized(weight.unsqueeze(0), scales.to(torch.bfloat16).unsqueeze(0))[0]


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite batched GPTQ equivalence result")
    capture = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    if capture["status"] != "capture_pass":
        raise RuntimeError("calibration capture gate did not pass")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    weight_map = checkpoint_weight_map(MODEL)
    rows = []
    total_official_seconds = 0.0
    total_batched_seconds = 0.0

    for layer_text, experts in lock["layers"].items():
        layer = int(layer_text)
        with safe_open(
            CALIBRATION_DIR / f"layer_{layer:02d}.safetensors", framework="pt", device="cpu"
        ) as handle:
            x = handle.get_tensor("moe_input")[experts].to(device).contiguous()
        original = load_weights(layer, experts, device, weight_map)
        z = original_down_inputs(x, original["gate"], original["up"])

        official = {kind: [] for kind in ("gate", "up", "down")}
        official_started = time.perf_counter()
        for index, _expert in enumerate(experts):
            official["gate"].append(official_pure_gptq_projection(original["gate"][index], x[index], GSQ_ROOT))
            official["up"].append(official_pure_gptq_projection(original["up"][index], x[index], GSQ_ROOT))
            official["down"].append(official_pure_gptq_projection(original["down"][index], z[index], GSQ_ROOT))
        torch.cuda.synchronize(device)
        official_seconds = time.perf_counter() - official_started
        total_official_seconds += official_seconds

        batched_started = time.perf_counter()
        accelerated = {kind: [] for kind in ("gate", "up", "down")}
        for index, _expert in enumerate(experts):
            accelerated["gate"].append(fastgrid_pure_gptq_projection(original["gate"][index], x[index], GSQ_ROOT))
            accelerated["up"].append(fastgrid_pure_gptq_projection(original["up"][index], x[index], GSQ_ROOT))
            accelerated["down"].append(fastgrid_pure_gptq_projection(original["down"][index], z[index], GSQ_ROOT))
        torch.cuda.synchronize(device)
        batched_seconds = time.perf_counter() - batched_started
        total_batched_seconds += batched_seconds

        batched_items = {
            kind: (
                torch.stack([item.weight for item in accelerated[kind]]),
                torch.stack([item.scales for item in accelerated[kind]]),
            )
            for kind in accelerated
        }
        for index, expert in enumerate(experts):
            for kind in ("gate", "up", "down"):
                batched_weight, batched_scales = batched_items[kind]
                official_item = official[kind][index]
                official_codes = codes(official_item.weight, official_item.scales)
                batched_codes = codes(batched_weight[index], batched_scales[index])
                scale_exact = torch.equal(
                    official_item.scales.to(torch.bfloat16).view(torch.uint16).cpu(),
                    batched_scales[index].view(torch.uint16).cpu(),
                )
                code_exact = torch.equal(official_codes.cpu(), batched_codes.cpu())
                rows.append({
                    "layer": layer, "expert": expert, "matrix": kind,
                    "codes_exact": code_exact, "bf16_scale_bits_exact": scale_exact,
                    "code_mismatches": int((official_codes != batched_codes).sum()),
                    "scale_bit_mismatches": int((
                        official_item.scales.to(torch.bfloat16).view(torch.uint16)
                        != batched_scales[index].view(torch.uint16)
                    ).sum()),
                })
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({
            "layer": layer, "experts": experts, "official_seconds": official_seconds,
            "batched_seconds": batched_seconds,
            "all_exact": all(row["codes_exact"] and row["bf16_scale_bits_exact"] for row in rows if row["layer"] == layer),
        }), flush=True)
        del x, z, original, official, accelerated, batched_items
        torch.cuda.empty_cache()

    passed = len(rows) == lock["required_independent_matrices"] and all(
        row["codes_exact"] and row["bf16_scale_bits_exact"] for row in rows
    )
    payload = {
        "kind": "qwen_gptq_bank_p0_batched_equivalence_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "equivalence_pass" if passed else "equivalence_fail",
        "inputs": {
            "lock_sha256": sha256(LOCK),
            "calibration_result_sha256": sha256(CALIBRATION_RESULT),
            "gsq_gptq_sha256": sha256(GSQ_ROOT / "src/prior/gptq.py"),
            "gsq_quant_sha256": sha256(GSQ_ROOT / "src/prior/quant.py"),
            "semantics_erratum_sha256": sha256(ROOT / "reports/qwen_gptq_bank/P0_GPTQ_SEMANTICS_ERRATUM.md"),
            "attempt_h_lock_sha256": sha256(ATTEMPT),
        },
        "matrices": rows,
        "summary": {
            "tested_matrices": len(rows), "all_codes_exact": all(row["codes_exact"] for row in rows),
            "all_bf16_scale_bits_exact": all(row["bf16_scale_bits_exact"] for row in rows),
            "total_code_mismatches": sum(row["code_mismatches"] for row in rows),
            "total_scale_bit_mismatches": sum(row["scale_bit_mismatches"] for row in rows),
            "official_seconds": total_official_seconds, "batched_seconds": total_batched_seconds,
            "speedup": total_official_seconds / total_batched_seconds,
        },
        "resources": {
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": peak_rss, "elapsed_seconds": time.perf_counter() - started,
        },
        "software": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__},
        "claim_boundary": "Implementation-equivalence gate only; no complete-bank or model-quality claim.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Qwen GPTQ Bank — batched GPTQ equivalence\n\n"
        f"Uitkomst: **{payload['status']}**.\n\n"
        f"Geteste matrices: {len(rows)}; codemismatches: {payload['summary']['total_code_mismatches']}; "
        f"BF16-scalebitmismatches: {payload['summary']['total_scale_bit_mismatches']}; "
        f"gemeten versnelling op de auditset: {payload['summary']['speedup']:.3f}×.\n\n"
        "Alleen bij een volledige exacte pass mag de gebatchte producer de bank maken.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], **payload["summary"],
    }, indent=2))
