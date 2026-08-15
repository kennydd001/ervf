from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import het_next_l0_ph0r3_common as common
import het_next_l0_ph0r3_intel as intel_base
import run_het_next_l0_ph0x_exploratory_real_projection as ph0x
import run_het_next_l0_ph0x_r2_exploratory_real_projection as r2


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r3_exploratory_real_projection"
RESULT = RUN / "ph0x_r3_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R3_EXPLORATORY_REAL_PROJECTION_FAIL_CLOSED_PREREGISTRATION_2026-08-13.md"

EXPECTED_FILES = {
    Path(common.__file__).resolve(): "899659ada099bd2efe1b95809a169b1f73b887ef31e1eb357d9f55f233121a46",
    Path(intel_base.__file__).resolve(): "002233f6d860015f9eadad613296c35b71a144cac6efaa5eaac5f2218c5eb004",
    Path(ph0x.__file__).resolve(): "e1429c0afe1d3ffbdafddaa375cc24c19955a6fce682845d924e0d707de5146b",
    Path(r2.__file__).resolve(): "f35e07f66092573845dd9a7b251e566ff82def2312d09e3f3d07c8ae161f4649",
}
EXPECTED_INTEL_SOURCE_SHA = "916474f85b9f077ca1acc203088d5da6c7d54b762b0764d90b6ae56eee579e61"
EXPECTED_CUDA_SOURCE_SHA = "942f181bb71eb7382f5f0e0398b4dd2ecba2563fb22348040f3c183ce37b2aca"
OLD_ATOMIC = "atomic_inc((volatile __global atomic_uint*)&count[row]);"
NEW_ATOMIC = "atomic_inc((volatile __global unsigned int*)&count[row]);"

if ph0x.INTEL_SOURCE.count(OLD_ATOMIC) != 1 or ph0x.INTEL_SOURCE.count(NEW_ATOMIC) != 0:
    raise RuntimeError("intel_atomic_repair_source_cardinality")
INTEL_SOURCE = ph0x.INTEL_SOURCE.replace(OLD_ATOMIC, NEW_ATOMIC)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dependency_gate() -> dict[str, str]:
    observed = {str(path): common.file_digest(path) for path in EXPECTED_FILES}
    if any(observed[str(path)] != expected for path, expected in EXPECTED_FILES.items()):
        raise RuntimeError("dependency_hash_drift")
    if INTEL_SOURCE.count(OLD_ATOMIC) != 0 or INTEL_SOURCE.count(NEW_ATOMIC) != 1:
        raise RuntimeError("intel_atomic_repair_result_cardinality")
    if sha(INTEL_SOURCE.encode()) != EXPECTED_INTEL_SOURCE_SHA:
        raise RuntimeError("intel_source_hash_drift")
    if sha(ph0x.CUDA_SOURCE.encode()) != EXPECTED_CUDA_SOURCE_SHA:
        raise RuntimeError("cuda_source_hash_drift")
    return observed


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    bindings = dependency_gate()
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_r3_exploratory_real_projection",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "prereg_sha256": common.file_digest(PREREG),
            "runner_sha256": common.file_digest(Path(__file__)),
            "dependencies": bindings,
            "intel_source_sha256": EXPECTED_INTEL_SOURCE_SHA,
            "cuda_source_sha256": EXPECTED_CUDA_SOURCE_SHA,
            "r2_prereg_sha256": "f369e8361bd2c78ba32ddf7c7952b836c8f263e6e5cae5ba68b8df1aa0aeeb6",
        },
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        record, evidence = common.build_record(source)
        safe = common.safe_check(record, input_bytes)
        controls = common.controls(record, input_bytes)[:8] + [r2.sensitivity_witness(record)]
        oracle = common.cpu_oracle(record, input_bytes)
        result.update(
            {
                "record_evidence": evidence,
                "safe_trace": safe["trace"],
                "controls": controls,
                "cpu_output_hex": oracle.tobytes().hex(),
                "cpu_output_sha256": sha(oracle.tobytes()),
            }
        )
        if len(controls) != 9 or not all(bool(row.get("pass")) for row in controls):
            raise RuntimeError("predevice_control_gate_negative")
        intel_base.SRC = INTEL_SOURCE
        intel = intel_base.run(record, input_bytes)
        intel_cmp = ph0x.compare(intel, oracle)
        result.update({"intel": intel, "intel_comparison": intel_cmp})
        if not (
            intel_cmp["different_words"] == 0
            and intel_cmp["counters_all_one"]
            and intel_cmp["sentinel_overwritten"]
            and bool(intel["ledger"][-1]["cleanup_complete"])
        ):
            raise RuntimeError("intel_gate_negative_before_nvidia")
        nvidia = ph0x.run_nvidia(record, input_bytes)
        nvidia_cmp = ph0x.compare(nvidia, oracle)
        result.update({"nvidia": nvidia, "nvidia_comparison": nvidia_cmp})
        gates = {
            "controls_all_pass": len(controls) == 9 and all(bool(row.get("pass")) for row in controls),
            "intel_exact": intel_cmp["different_words"] == 0,
            "nvidia_exact": nvidia_cmp["different_words"] == 0,
            "intel_counters": intel_cmp["counters_all_one"],
            "nvidia_counters": nvidia_cmp["counters_all_one"],
            "intel_sentinel": intel_cmp["sentinel_overwritten"],
            "nvidia_sentinel": nvidia_cmp["sentinel_overwritten"],
            "intel_cleanup": bool(intel["ledger"][-1]["cleanup_complete"]),
            "nvidia_cleanup": bool(nvidia["ledger"][-1]["cleanup_complete"]),
            "distinct_pci": intel["identity"]["pci"] != nvidia["identity"]["pci"],
        }
        result["gates"] = gates
        result["positive"] = all(gates.values())
        result["status"] = (
            "exploratory_single_real_projection_positive"
            if result["positive"]
            else "exploratory_single_real_projection_negative"
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update(
            {
                "positive": False,
                "status": "exploratory_single_real_projection_failure",
                "error": error,
                "traceback": traceback.format_exc(),
            }
        )
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["claim_boundary"] = (
        "One real projection/input, exploratory only; no full expert/layer/model/performance/"
        "concurrency/deployment/novelty/breakthrough claim."
    )
    common.write_atomic_new(RESULT, common.canonical(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "positive": result["positive"],
                "gates": result.get("gates"),
                "error": error,
            },
            indent=2,
        )
    )
    return 0 if result.get("positive") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
