from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
SOURCE_PATH = Path(__file__).with_name("run_p6a_end_to_end_decode.py")
PREREG = R / "P7D_EXACT_CE_REPAIR_PREREGISTRATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_namespace(variant: str) -> dict:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    required = {
        "        state_host = self.embedding(int(token))\n        wall_start = time.perf_counter_ns()":
            "        wall_start = time.perf_counter_ns()\n        state_host = self.embedding(int(token))",
        '            "labels": len(domain_ce), "next_token_cross_entropy": domain_mean_ce,':
            '            "labels": len(domain_ce), "cross_entropies": domain_ce, "next_token_cross_entropy": domain_mean_ce,',
        '            "labels": len(all_ce), "next_token_cross_entropy": aggregate_ce,':
            '            "labels": len(all_ce), "cross_entropies": all_ce, "next_token_cross_entropy": aggregate_ce,',
    }
    if variant == "ervf":
        required.update({
            "module = cp.RawModule(code=CUDA_SOURCE,": "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,",
            '"q8_gemv", "rmsnorm"': '"q8_gemv", "q8_ervf16", "rmsnorm"',
            '"attention_values", "residual_add", "q5_gate_up_n", "swiglu_n", "q5_down_n",':
                '"attention_values", "residual_add", "q5_gate_up_n", "q5_gate_up_ervf16", "swiglu_n", "q5_down_n", "q5_down_ervf16",',
            'self.k["q8_gemv"]((record["rows"],), (256,), (':
                'self.k["q8_ervf16"](((record["rows"] + 15) // 16,), (256,), (',
            'self.k["q5_gate_up_n"]((count * 1536,), (256,), (':
                'self.k["q5_gate_up_ervf16"](((count * 1536 + 15) // 16,), (256,), (',
            'self.k["q5_down_n"]((count * 2048,), (256,), (':
                'self.k["q5_down_ervf16"](((count * 2048 + 15) // 16,), (256,), (',
        })
    for old, new in required.items():
        if old not in source:
            raise RuntimeError(f"P7D transform target missing: {old}")
        source = source.replace(old, new)
    namespace = {"__name__": "p7d_runtime_library", "__file__": str(SOURCE_PATH), "ERVF_SOURCE": ERVF_SOURCE}
    exec(compile(source, str(SOURCE_PATH), "exec"), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "ervf"), required=True)
    args = parser.parse_args()
    output = R / f"p7d_exact_ce_{args.variant}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ns = runtime_namespace(args.variant)
    lock = json.loads((R / "p6a_end_to_end_input_lock.json").read_text(encoding="utf-8"))
    runtime = ns["Runtime"](lock)
    tensors = ns["load_file"](ns["P0C_DATA"])
    phases = {}
    for split, teacher_path in (("validation", ns["P0C_VALIDATION"]), ("test", ns["P0C_TEST"])):
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        phases[split] = ns["quality_phase"](runtime, split, tensors, teacher)
    result = {
        "kind": "streamq5_moe_p7d_exact_ce_repair", "variant": args.variant,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "source_sha256": sha256(SOURCE_PATH), "base_input_lock_sha256": sha256(R / "p6a_end_to_end_input_lock.json"),
        "physical": runtime.physical(), "phases": phases,
        "claim_boundary": "Evidence-retention repair only; timings are excluded from performance claims.",
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "variant": args.variant, "validation_labels": phases["validation"]["aggregate"]["labels"], "test_labels": phases["test"]["aggregate"]["labels"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
