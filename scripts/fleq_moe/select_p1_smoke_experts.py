from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import torch
from safetensors import safe_open

from moe_lab.fleq_moe.expert_quant import select_most_frequent_experts
from moe_lab.reporting import ROOT


CAPTURE = ROOT / "reports/runs/rsiv_moe/p1c_qwen_validation.safetensors"
PREREGISTRATION = ROOT / "reports/fleq_moe/P1_QWEN_EXPERT_STREAMED_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
LAYERS = (0, 47)
CONTEXT_TOKENS = 1152


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    rows = {}
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        for layer in LAYERS:
            ids = handle.get_tensor(f"layer_{layer:02d}_router_ids").long()
            calibration_ids = ids[:CONTEXT_TOKENS]
            counts = torch.bincount(calibration_ids.reshape(-1), minlength=128)
            selected = select_most_frequent_experts(calibration_ids, 8)
            rows[str(layer)] = {
                "selected_experts": selected,
                "calibration_counts": {str(expert): int(counts[expert]) for expert in selected},
            }
    payload = {
        "kind": "fleq_moe_p1_smoke_expert_selection_lock",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "validation_capture_sha256": sha256(CAPTURE),
        "selection_rule": "top 8 context-0 counts; ties by ascending expert ID",
        "context_1_unopened_for_metrics": True,
        "layers": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

