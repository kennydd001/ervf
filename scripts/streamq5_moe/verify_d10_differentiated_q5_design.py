#!/usr/bin/env python3
"""CPU-only verifier for the D10 differentiated-Q5 design audit."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
OUT = R / "d10_differentiated_q5_design_verification.json"
P0_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_p0_physical_host_bank.py"
D7_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d7_staged_exact_q5_plane.py"
D9_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d9_capacity_aware_bank_bridge.py"
P4D = R / "p4d_route_capture_result.json"
D9 = R / "port80b_d9_capacity_aware_bank_bridge.json"
P4D_LAYER0 = R / "p4d_route_layers" / "layer_00.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_round(value: float) -> tuple[float, int]:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    lsb = (bits >> 16) & 1
    word = ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF
    rounded = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return rounded, word


checks: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: object = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


p0_source = P0_RUNNER.read_text(encoding="utf-8")
d7_source = D7_RUNNER.read_text(encoding="utf-8")
d9_source = D9_RUNNER.read_text(encoding="utf-8")
p4d = json.loads(P4D.read_text(encoding="utf-8"))
d9 = json.loads(D9.read_text(encoding="utf-8"))
p4d_layer0 = json.loads(P4D_LAYER0.read_text(encoding="utf-8"))

check("p0_payload_is_invariant", "bytes([0x55]) * CODE_BYTES" in p0_source
      and "0x3C00" in p0_source)
check("d7_compute_ignores_headers", "bank + base + 64" in d7_source
      and "const unsigned short* scales" in d7_source)
check("d9_route_integrity_is_separate_byte_oracle", "full_verify" in d9_source
      and d9["integrity"]["negative_controls"]["wrong_expert"]["detected_mismatches"] > 0
      and d9["integrity"]["negative_controls"]["wrong_layer"]["detected_mismatches"] > 0)

# One exact BF16 identifier per (layer, expert). BF16 numbers in [1,2) have
# only 128 distinct mantissas, so one scalar cannot encode 24,576 identities.
# Three radix-32 BF16 digits can. Keep the original 2^-7 scale magnitude and
# use scale=2^-7*(1+digit/64), which is exactly BF16 for digit 0..31.
words: dict[tuple[int, int], tuple[int, int, int]] = {}
decoded: dict[tuple[int, int, int], tuple[int, int]] = {}
exact = True
for layer in range(48):
    for expert in range(512):
        identity = layer * 512 + expert
        digits = ((identity >> 10) & 31, (identity >> 5) & 31, identity & 31)
        stored: list[int] = []
        for digit in digits:
            value = (2.0 ** -7) * (1.0 + digit / 64.0)
            rounded, word = bf16_round(value)
            exact &= rounded == value
            stored.append(word)
        key = tuple(stored)
        words[(layer, expert)] = key
        if key in decoded:
            exact = False
        decoded[key] = (layer, expert)

check("three_bf16_digit_code_exact_and_injective", exact and len(words) == 48 * 512
      and len(decoded) == 48 * 512, {"identities": len(words), "bytes_if_materialized": len(words) * 3 * 2})
check("single_bf16_scalar_is_insufficient", 48 * 512 > 128,
      "A single normal BF16 in [1,2) has only 128 exact mantissa values; use three radix-32 digits.")

# The oracle uses UP rows 0..2, group 0, and one-hot x[0]=1. With the frozen
# 0x55 code stream the first decoded Q5 code is +6. The patched scales produce
# exact nonzero, distinct row outputs after BF16 rounding.
outputs: set[tuple[int, int, int]] = set()
output_exact = True
for stored in words.values():
    rows = []
    for word in stored:
        scale = struct.unpack("<f", struct.pack("<I", word << 16))[0]
        rounded, out_word = bf16_round(6.0 * scale)
        output_exact &= rounded != 0.0
        rows.append(out_word)
    outputs.add(tuple(rows))
check("one_hot_up_canary_is_numerically_injective", output_exact and len(outputs) == 48 * 512,
      {"distinct_output_triplets": len(outputs)})

# P4D is a valid real route capture for a different model geometry, not a
# natural/representative Qwen3-Coder-Next trace.
check("p4d_capture_is_complete", p4d.get("status") == "route_capture_complete"
      and p4d.get("layers") == 48 and p4d.get("tokens_per_domain") == 1024
      and len(p4d.get("domains", [])) == 5)
check("p4d_target_geometry_mismatch_identified",
      p4d.get("model_variant") == "q5_experts_int8_trunk"
      and all(len(counts) == 128 for counts in p4d_layer0["domain_counts"].values())
      and d9["physical"]["registered_experts_per_layer"] + d9["physical"]["cold_escape_experts_per_layer"] == 512,
      {"p4d_experts": 128, "p4d_top_k": 8, "d9_experts": 512, "d9_top_k": 10})

# Wrong-pointer controls must derive expected values from the intended route,
# not from the candidate pointer/header. Verify the mathematical oracle changes.
intended = words[(0, 353)]
wrong_expert = words[(0, 354)]
wrong_layer = words[(1, 353)]
check("wrong_expert_numeric_control_is_detectable", intended != wrong_expert)
check("wrong_layer_numeric_control_is_detectable", intended != wrong_layer)

failed = [row for row in checks if not row["pass"]]
result = {
    "kind": "d10_differentiated_q5_design_cpu_verification",
    "cpu_only": True,
    "gpu_runtime_imported": False,
    "overall_pass": not failed,
    "checks_passed": len(checks) - len(failed),
    "checks_total": len(checks),
    "failures": failed,
    "inputs": {
        "p0_runner_sha256": sha256(P0_RUNNER),
        "d7_runner_sha256": sha256(D7_RUNNER),
        "d9_runner_sha256": sha256(D9_RUNNER),
        "p4d_capture_sha256": sha256(P4D),
        "p4d_layer0_report_sha256": sha256(P4D_LAYER0),
        "d9_result_sha256": sha256(D9),
    },
    "design": {
        "new_large_bank_required": False,
        "existing_p0_bank_mode": "read_only",
        "patch_location": "D9 active HBM staging plane after byte-integrity verification and before Q5 compute",
        "patched_words_per_selected_expert_record": 3,
        "patch_projection": "up",
        "patch_rows": [0, 1, 2],
        "patch_group": 0,
        "input_oracle": "x[0]=1 and all other x=0",
        "identity": "i=layer*512+expert; radix-32 digits d2,d1,d0; scale[j]=BF16(2^-7*(1+d[j]/64))",
        "expected_canary": "BF16(6*scale[j]) for UP output rows 0,1,2",
        "maximum_materialized_codebook_bytes": 48 * 512 * 3 * 2,
    },
    "claim_boundary": (
        "This verifies design arithmetic and local provenance only. It does not execute GPU code, "
        "modify the bank, validate throughput, create a real checkpoint or make P4D representative of Qwen3-Coder-Next."
    ),
    "checks": checks,
}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ("overall_pass", "checks_passed", "checks_total", "failures")}, indent=2))
raise SystemExit(0 if result["overall_pass"] else 1)
