from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


source_path = Path(__file__).with_name("run_p7c_ervf_end_to_end.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P7C_ERVF_END_TO_END_PREREGISTRATION.md": "P8A2_Q5_WIDTH8_END_TO_END_PREREGISTRATION.md",
    "p7c_ervf_end_to_end_input_lock.json": "p8a2_q5_width8_end_to_end_input_lock.json",
    "p7c_ervf_end_to_end_evaluator_lock.json": "p8a2_q5_width8_end_to_end_evaluator_lock.json",
    "p7c_ervf_end_to_end_smoke.json": "p8a2_q5_width8_end_to_end_smoke.json",
    "p7c_ervf_end_to_end_validation.json": "p8a2_q5_width8_end_to_end_validation.json",
    "p7c_ervf_end_to_end_{args.phase}.json": "p8a2_q5_width8_end_to_end_{args.phase}.json",
    "P7C_ERVF_END_TO_END_{args.phase.upper()}.md": "P8A2_Q5_WIDTH8_END_TO_END_{args.phase.upper()}.md",
    "p7c_ervf_smoke_pass": "p8a2_q5_width8_smoke_pass",
    "p7c_ervf_smoke_fail": "p8a2_q5_width8_smoke_fail",
    "p7c_ervf_validation_pass_test_authorized": "p8a2_q5_width8_validation_pass_test_authorized",
    "p7c_ervf_validation_closed_test_unopened": "p8a2_q5_width8_validation_closed_test_unopened",
    "p7c_ervf_end_to_end_pass": "p8a2_q5_width8_end_to_end_pass",
    "p7c_ervf_end_to_end_closed": "p8a2_q5_width8_end_to_end_closed",
    "streamq5_moe_p7c_ervf_end_to_end_smoke": "streamq5_moe_p8a2_q5_width8_end_to_end_smoke",
    "streamq5_moe_p7c_ervf_physical_end_to_end_decode": "streamq5_moe_p8a2_q5_width8_physical_end_to_end_decode",
    "P7C input provenance mismatch": "P8A2 input provenance mismatch",
    "P7C evaluator provenance mismatch": "P8A2 evaluator provenance mismatch",
    "P7C smoke pass required": "P8A2 smoke pass required",
    "P7C test not authorized": "P8A2 test not authorized",
    "P7C phase output": "P8A2 phase output",
    "# P7C ERVF fysieke end-to-end decode": "# P8A2 Q5-width-8 fysieke end-to-end decode",
    "p7b_ervf_kernel.json": "p8a_projection_adaptive_ervf.json",
    "p7b_result_sha256": "p8a_result_sha256",
    "P7C P7B provenance mismatch": "P8A2 P8A provenance mismatch",
    "q5_gate_up_ervf16": "q5_gate_up_ervf8",
    "q5_down_ervf16": "q5_down_ervf8",
    "((count * 1536 + 15) // 16,)": "((count * 1536 + 31) // 32,)",
    "((count * 2048 + 15) // 16,)": "((count * 2048 + 31) // 32,)",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"P8A2 source transform target missing: {old}")
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__})
