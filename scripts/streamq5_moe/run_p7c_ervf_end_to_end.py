from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE


source_path = Path(__file__).with_name("run_p6a_end_to_end_decode.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P6A_END_TO_END_DECODE_PREREGISTRATION.md": "P7C_ERVF_END_TO_END_PREREGISTRATION.md",
    "p6a_end_to_end_input_lock.json": "p7c_ervf_end_to_end_input_lock.json",
    "p6a_end_to_end_evaluator_lock.json": "p7c_ervf_end_to_end_evaluator_lock.json",
    "p6a_end_to_end_smoke.json": "p7c_ervf_end_to_end_smoke.json",
    "p6a_end_to_end_validation.json": "p7c_ervf_end_to_end_validation.json",
    "p6a_end_to_end_{args.phase}.json": "p7c_ervf_end_to_end_{args.phase}.json",
    "P6A_END_TO_END_{args.phase.upper()}.md": "P7C_ERVF_END_TO_END_{args.phase.upper()}.md",
    "p6a_smoke_pass": "p7c_ervf_smoke_pass",
    "p6a_smoke_fail": "p7c_ervf_smoke_fail",
    "p6a_validation_pass_test_authorized": "p7c_ervf_validation_pass_test_authorized",
    "p6a_validation_closed_test_unopened": "p7c_ervf_validation_closed_test_unopened",
    "p6a_end_to_end_eureka_pass": "p7c_ervf_end_to_end_pass",
    "p6a_end_to_end_closed": "p7c_ervf_end_to_end_closed",
    "streamq5_moe_p6a_end_to_end_smoke": "streamq5_moe_p7c_ervf_end_to_end_smoke",
    "streamq5_moe_p6a_physical_end_to_end_decode": "streamq5_moe_p7c_ervf_physical_end_to_end_decode",
    "P6A input provenance mismatch": "P7C input provenance mismatch",
    "P6A evaluator provenance mismatch": "P7C evaluator provenance mismatch",
    "P6A smoke pass required": "P7C smoke pass required",
    "P6A test not authorized": "P7C test not authorized",
    "P6A phase output": "P7C phase output",
    "# P6A fysieke end-to-end decode": "# P7C ERVF fysieke end-to-end decode",
    "        state_host = self.embedding(int(token))\n        wall_start = time.perf_counter_ns()":
        "        wall_start = time.perf_counter_ns()\n        state_host = self.embedding(int(token))",
    "    lock = json.loads(INPUT_LOCK.read_text(encoding=\"utf-8\"))":
        "    repair_lock = json.loads(INPUT_LOCK.read_text(encoding=\"utf-8\"))\n"
        "    base_lock_path = R / \"p6a_end_to_end_input_lock.json\"\n"
        "    if sha256(base_lock_path) != repair_lock[\"base_input_lock_sha256\"]:\n"
        "        raise ValueError(\"P7C base input-lock provenance mismatch\")\n"
        "    if sha256(R / \"p7b_ervf_kernel.json\") != repair_lock[\"p7b_result_sha256\"]:\n"
        "        raise ValueError(\"P7C P7B provenance mismatch\")\n"
        "    lock = json.loads(base_lock_path.read_text(encoding=\"utf-8\"))\n"
        "    lock[\"preregistration_sha256\"] = repair_lock[\"preregistration_sha256\"]",
    "module = cp.RawModule(code=CUDA_SOURCE,": "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,",
    "\"q8_gemv\", \"rmsnorm\"": "\"q8_gemv\", \"q8_ervf16\", \"rmsnorm\"",
    "\"attention_values\", \"residual_add\", \"q5_gate_up_n\", \"swiglu_n\", \"q5_down_n\",":
        "\"attention_values\", \"residual_add\", \"q5_gate_up_n\", \"q5_gate_up_ervf16\", \"swiglu_n\", \"q5_down_n\", \"q5_down_ervf16\",",
    "self.k[\"q8_gemv\"]((record[\"rows\"],), (256,), (":
        "self.k[\"q8_ervf16\"](((record[\"rows\"] + 15) // 16,), (256,), (",
    "self.k[\"q5_gate_up_n\"]((count * 1536,), (256,), (":
        "self.k[\"q5_gate_up_ervf16\"](((count * 1536 + 15) // 16,), (256,), (",
    "self.k[\"q5_down_n\"]((count * 2048,), (256,), (":
        "self.k[\"q5_down_ervf16\"](((count * 2048 + 15) // 16,), (256,), (",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"P7C source transform target missing: {old}")
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})
