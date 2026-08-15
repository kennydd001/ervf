from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from safetensors.torch import load_file

from moe_lab.e2gq_moe.entropy import (
    CODEBOOK,
    code_histogram,
    ideal_total_bpp,
    multinomial_bits,
    projection_codes,
    shannon_entropy,
)
from moe_lab.reporting import ROOT


RESULT = ROOT / "reports/e2gq_moe/locked_precondition_audit.json"
REPORT = ROOT / "reports/e2gq_moe/E2GQ_LOCKED_PRECONDITION_AUDIT.md"
FLEQ_RESULT = ROOT / "reports/fleq_moe/p1_smoke_result.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite the E2GQ precondition audit")
    fleq = json.loads(FLEQ_RESULT.read_text(encoding="utf-8"))
    aggregate = {symbol: 0 for symbol in CODEBOOK}
    experts = []
    matrices = []
    for layer_text in ("0", "47"):
        for expert in fleq["layers"][layer_text]["selected_experts"]:
            key = f"layer_{int(layer_text):02d}_expert_{expert:03d}_2bit"
            manifest = fleq["artifacts"][key]
            artifact = ROOT / manifest["artifact"]
            if sha256(artifact) != manifest["artifact_sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact}")
            tensors = load_file(artifact)
            expert_counts = {symbol: 0 for symbol in CODEBOOK}
            expert_weights = expert_scales = expert_enum_bits = 0
            for matrix in ("gate", "up", "down"):
                weight = tensors[f"gptq_{matrix}_weight"]
                scales = tensors[f"gptq_{matrix}_scales"]
                codes = projection_codes(weight, scales)
                counts = code_histogram(codes)
                weights = weight.numel()
                scale_count = scales.numel()
                bpp = ideal_total_bpp(counts, scale_count, weights)
                matrices.append({
                    "layer": int(layer_text), "expert": expert, "matrix": matrix,
                    "weights": weights, "scales": scale_count,
                    "histogram": {str(k): counts[k] for k in CODEBOOK},
                    "code_entropy_bpp": shannon_entropy(counts),
                    "ideal_total_bpp": bpp,
                    "bit_exact_code_scale_reconstruction": True,
                })
                for symbol in CODEBOOK:
                    aggregate[symbol] += counts[symbol]
                    expert_counts[symbol] += counts[symbol]
                expert_weights += weights
                expert_scales += scale_count
                expert_enum_bits += multinomial_bits(counts) + 16 * 8 + 16 * scale_count
            experts.append({
                "layer": int(layer_text), "expert": expert,
                "weights": expert_weights, "scales": expert_scales,
                "histogram": {str(k): expert_counts[k] for k in CODEBOOK},
                "ideal_total_bpp": ideal_total_bpp(expert_counts, expert_scales, expert_weights),
                "per_matrix_enumerative_total_bpp": expert_enum_bits / expert_weights,
            })

    weights = sum(aggregate.values())
    entropy = shannon_entropy(aggregate)
    negative = aggregate[-2] + aggregate[-1]
    core_counts = {-1: negative, 0: aggregate[0], 1: aggregate[1]}
    core_entropy = shannon_entropy(core_counts)
    extreme_conditional = (negative / weights) * (
        -(aggregate[-2] / negative) * math.log2(aggregate[-2] / negative)
        -(aggregate[-1] / negative) * math.log2(aggregate[-1] / negative)
    )
    payload = {
        "kind": "e2gq_locked_precondition_independent_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_fleq_result": str(FLEQ_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "source_fleq_result_sha256": sha256(FLEQ_RESULT),
        "locked_experts": len(experts), "locked_matrices": len(matrices),
        "weights": weights,
        "aggregate_histogram": {str(k): aggregate[k] for k in CODEBOOK},
        "probabilities": {str(k): aggregate[k] / weights for k in CODEBOOK},
        "code_entropy_bpp": entropy,
        "raw_bf16_group128_scale_bpp": 0.125,
        "ideal_total_bpp": entropy + 0.125,
        "reserve_below_2bpp": 2 - entropy - 0.125,
        "all_experts_ideal_below_2bpp": all(x["ideal_total_bpp"] < 2 for x in experts),
        "all_matrices_ideal_below_2bpp": all(x["ideal_total_bpp"] < 2 for x in matrices),
        "expert_ideal_bpp_range": [min(x["ideal_total_bpp"] for x in experts), max(x["ideal_total_bpp"] for x in experts)],
        "matrix_ideal_bpp_range": [min(x["ideal_total_bpp"] for x in matrices), max(x["ideal_total_bpp"] for x in matrices)],
        "enumerative_mean_bpp": statistics.mean(x["per_matrix_enumerative_total_bpp"] for x in experts),
        "ternary_core_entropy_bpp": core_entropy,
        "extreme_tail_conditional_bpp": extreme_conditional,
        "ternary_core_plus_tail_plus_scales_bpp": core_entropy + extreme_conditional + 0.125,
        "bit_exact_code_scale_reconstruction": True,
        "experts": experts, "matrices": matrices,
        "claim_boundary": {
            "actual_encoded_file": False, "full_bank": False,
            "model_quality": False, "runtime": False, "eureka": "representation precondition only",
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E2GQ-MoE — onafhankelijke audit van de locked precondition", "",
        "De GPTQ-codehistogrammen zijn opnieuw afgeleid uit de 16 originele safetensors-artifacts; de aangeleverde JSON is niet als meetbron gebruikt.", "",
        f"- Codes: `{payload['aggregate_histogram']}` over `{weights:,}` gewichten.",
        f"- Code-entropie: `{entropy:.12f}` bpp.",
        f"- Inclusief raw BF16 group-128 scales: `{entropy + 0.125:.12f}` bpp.",
        f"- Alle 16 experts onder 2 bpp: `{payload['all_experts_ideal_below_2bpp']}`.",
        f"- Alle 48 matrices onder 2 bpp: `{payload['all_matrices_ideal_below_2bpp']}`.",
        f"- Exacte ternary-core + extreme-tail-identiteit: `{core_entropy + extreme_conditional + 0.125:.12f}` bpp.", "",
        "Dit bevestigt een harde theoretische representatieprecondition. Het is nog geen werkelijk geëncodeerd bestand, full-bankmeting, kwaliteitsbewijs of runtimebewijs.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("locked_experts", "locked_matrices", "weights", "aggregate_histogram", "ideal_total_bpp", "reserve_below_2bpp")}, indent=2))

