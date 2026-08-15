from pathlib import Path

source_path = Path(__file__).with_name("run_p6a_end_to_end_decode.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P6A_END_TO_END_DECODE_PREREGISTRATION.md": "P6B_STRICT_END_TO_END_REPLICATION_PREREGISTRATION.md",
    "p6a_end_to_end_input_lock.json": "p6b_strict_end_to_end_input_lock.json",
    "p6a_end_to_end_evaluator_lock.json": "p6b_strict_end_to_end_evaluator_lock.json",
    "p6a_end_to_end_smoke.json": "p6b_strict_end_to_end_smoke.json",
    "p6a_end_to_end_validation.json": "p6b_strict_end_to_end_validation.json",
    "p6a_end_to_end_{args.phase}.json": "p6b_strict_end_to_end_{args.phase}.json",
    "P6A_END_TO_END_{args.phase.upper()}.md": "P6B_STRICT_END_TO_END_{args.phase.upper()}.md",
    "p6a_smoke_pass": "p6b_smoke_pass",
    "p6a_smoke_fail": "p6b_smoke_fail",
    "p6a_validation_pass_test_authorized": "p6b_validation_pass_test_authorized",
    "p6a_validation_closed_test_unopened": "p6b_validation_closed_test_unopened",
    "p6a_end_to_end_eureka_pass": "p6b_strict_end_to_end_eureka_pass",
    "p6a_end_to_end_closed": "p6b_strict_end_to_end_closed",
    "streamq5_moe_p6a_end_to_end_smoke": "streamq5_moe_p6b_strict_end_to_end_smoke",
    "streamq5_moe_p6a_physical_end_to_end_decode": "streamq5_moe_p6b_strict_physical_end_to_end_decode",
    "P6A input provenance mismatch": "P6B input provenance mismatch",
    "P6A evaluator provenance mismatch": "P6B evaluator provenance mismatch",
    "P6A smoke pass required": "P6B smoke pass required",
    "P6A test not authorized": "P6B test not authorized",
    "P6A phase output": "P6B phase output",
    "# P6A fysieke end-to-end decode": "# P6B strikte fysieke end-to-end decode",
    "    lock = json.loads(INPUT_LOCK.read_text(encoding=\"utf-8\"))":
        "    repair_lock = json.loads(INPUT_LOCK.read_text(encoding=\"utf-8\"))\n"
        "    base_lock_path = R / \"p6a_end_to_end_input_lock.json\"\n"
        "    if sha256(base_lock_path) != repair_lock[\"base_input_lock_sha256\"]:\n"
        "        raise ValueError(\"P6B base input-lock provenance mismatch\")\n"
        "    lock = json.loads(base_lock_path.read_text(encoding=\"utf-8\"))\n"
        "    lock[\"preregistration_sha256\"] = repair_lock[\"preregistration_sha256\"]",
    "        state_host = self.embedding(int(token))\n        wall_start = time.perf_counter_ns()":
        "        wall_start = time.perf_counter_ns()\n        state_host = self.embedding(int(token))",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"P6B source transform target missing: {old}")
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__})
