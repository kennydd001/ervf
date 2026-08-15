from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
P0_EVALUATOR = ROOT / "scripts/streamq5_moe/run_p0_model_quality.py"
OUT = ROOT / "reports/streamq5_moe/q5_scale_semantics_audit.json"
REPORT = ROOT / "reports/streamq5_moe/Q5_SCALE_SEMANTICS_AUDIT.md"
CASES = ((0, 0, "gate"), (0, 127, "down"), (24, 63, "up"), (47, 0, "down"), (47, 127, "gate"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite Q5 scale semantics audit")
    weight_map = checkpoint_weight_map(MODEL)
    rows = []
    for layer, expert, projection in CASES:
        name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        weight = load_checkpoint_tensors(MODEL, [name], weight_map)[name].cuda()
        shape = weight.shape
        work = weight.float().reshape(shape[0], shape[1] // 128, 128)
        scale_fp32 = torch.where(work.abs().amax(-1, keepdim=True) > 0, work.abs().amax(-1, keepdim=True) / 15, torch.ones_like(work[..., :1]))
        codes = torch.round(work / scale_fp32).clamp(-15, 15)
        evaluator_dequant = (codes * scale_fp32).reshape(shape).to(torch.bfloat16)
        stored_scale = scale_fp32.to(torch.bfloat16)
        physical_dequant = (codes * stored_scale.float()).reshape(shape).to(torch.bfloat16)
        different = evaluator_dequant.view(torch.uint16) != physical_dequant.view(torch.uint16)
        delta = evaluator_dequant.float() - physical_dequant.float()
        rows.append({
            "layer": layer, "expert": expert, "projection": projection,
            "weights": weight.numel(), "different_bf16_values": int(different.sum()),
            "different_fraction": float(different.float().mean()),
            "max_abs": float(delta.abs().max()), "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(evaluator_dequant.float()).clamp_min(1e-30)),
            "codes_min": int(codes.min()), "codes_max": int(codes.max()),
        })
    total_values = sum(row["weights"] for row in rows); total_different = sum(row["different_bf16_values"] for row in rows)
    exact = total_different == 0
    payload = {
        "kind": "streamq5_moe_q5_scale_semantics_audit", "status": "scale_semantics_exact" if exact else "scale_semantics_mismatch",
        "inputs": {"p0_evaluator_sha256": sha256(P0_EVALUATOR), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "cases": rows, "aggregate": {"weights": total_values, "different_bf16_values": total_different, "different_fraction": total_different / total_values},
        "interpretation": "Compares the implemented FP32-scale-then-BF16 path with the preregistered BF16-stored-scale-then-BF16 path.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(f"# STREAMQ5 Q5-schaalsemantiek\n\nUitkomst: **{payload['status']}**. {total_different:,}/{total_values:,} BF16-waarden verschillen ({total_different / total_values:.4%}).\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"], "cases": rows}, indent=2))
