from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from moe_lab.fleq_moe.expert_quant import official_gptq_projection
from moe_lab.qwen_gptq_bank import codes_from_quantized, official_pure_gptq_projection
from moe_lab.reporting import ROOT


GSQ = ROOT / "third_party/GSQ"
HELPER = ROOT / "src/moe_lab/fleq_moe/expert_quant.py"
UPSTREAM = GSQ / "src/prior/gptq.py"
ERRATUM = ROOT / "reports/qwen_gptq_bank/P0_GPTQ_SEMANTICS_ERRATUM.md"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_gptq_semantics_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codes(item):
    return codes_from_quantized(
        item.weight.to(torch.bfloat16).unsqueeze(0),
        item.scales.to(torch.bfloat16).unsqueeze(0),
    )[0]


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite GPTQ semantics audit")
    helper_text = HELPER.read_text(encoding="utf-8")
    upstream_text = UPSTREAM.read_text(encoding="utf-8")
    historical_name = "fleq_projection"
    neutral_name = "expert_projection"
    markers = ("q_proj", "k_proj", "in_proj_qkv")
    static = {
        "helper_uses_historical_name": f'GPTQ(layer, "{historical_name}"' in helper_text,
        "upstream_has_name_gated_branch": 'if "q_proj" in self.name or "k_proj" in self.name or "in_proj_qkv" in self.name' in upstream_text,
        "historical_name_matches_q_proj": "q_proj" in historical_name,
        "neutral_name_matches_no_marker": not any(marker in neutral_name for marker in markers),
        "upstream_branch_runs_2000_epochs": "num_epochs = 2000" in upstream_text,
    }

    torch.manual_seed(260811)
    weight = (torch.randn(8, 128) / 20).to(torch.bfloat16)
    calibration = torch.randn(128, 128).to(torch.bfloat16)
    historical = official_gptq_projection(weight, calibration, GSQ)
    pure = official_pure_gptq_projection(weight, calibration, GSQ)
    historical_codes, pure_codes = codes(historical), codes(pure)
    behavioral = {
        "integer_code_mismatches": int((historical_codes != pure_codes).sum()),
        "bf16_scale_bit_mismatches": int((
            historical.scales.to(torch.bfloat16).view(torch.uint16)
            != pure.scales.to(torch.bfloat16).view(torch.uint16)
        ).sum()),
        "outputs_differ": not torch.equal(historical_codes, pure_codes),
    }
    passed = all(static.values()) and behavioral["outputs_differ"]
    payload = {
        "kind": "qwen_gptq_bank_p0_gptq_semantics_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "historical_name_collision_confirmed" if passed else "audit_failed",
        "static_control_flow": static, "behavioral_probe": behavioral,
        "inputs": {
            "helper_sha256": sha256(HELPER), "upstream_gptq_sha256": sha256(UPSTREAM),
            "erratum_sha256": sha256(ERRATUM), "seed": 260811,
            "probe_weight_shape": [8, 128], "probe_calibration_shape": [128, 128],
        },
        "claim_boundary": "Confirms the historical helper name collision; does not assess model quality.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
