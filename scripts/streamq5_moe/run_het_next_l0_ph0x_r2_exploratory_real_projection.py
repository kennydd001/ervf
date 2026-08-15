from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import het_next_l0_ph0r3_common as common
import het_next_l0_ph0r3_intel as intel_base
import run_het_next_l0_ph0x_exploratory_real_projection as ph0x


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r2_exploratory_real_projection"
RESULT = RUN / "ph0x_r2_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R2_EXPLORATORY_REAL_PROJECTION_REPAIR_PREREGISTRATION_2026-08-13.md"

EXPECTED_ACTIVATION_SHA = "2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561"
EXPECTED_ORIGINAL_SHA = "98fac647d0adc50536d5b397b1974ac237ec14a818ac4ec287760dbab312400b"
EXPECTED_MUTATED_SHA = "3571cfc8dbc22de68d5b216fa5766b3bc0036e745062dda3d645fc1b1c019910"


INTEL_SOURCE = ph0x.INTEL_SOURCE.replace(
    "atomic_inc((volatile __global atomic_uint*)&count[row]);",
    "atomic_inc((volatile __global unsigned int*)&count[row]);",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sensitivity_witness(record: bytes) -> dict[str, object]:
    codes, scales = common.split_record(record)
    fields = common.unpack_fields(codes)
    flat = fields.reshape(-1)
    index = int(np.flatnonzero(flat != 15)[0])
    stored = int(flat[index])
    step = stored - 1 if stored > 15 else stored + 1
    activation = np.zeros(common.COLS, dtype="<u2")
    activation[0] = 0x3B80
    original = np.zeros(common.ROWS, dtype="<u2")
    mutated = np.zeros(common.ROWS, dtype="<u2")
    scale_word = int(np.frombuffer(scales, "<u2")[0])
    scale = common.bf16_to_f32(np.asarray([scale_word], dtype=np.uint16))[0]
    q0, q1 = stored - 15, step - 15
    original_weight = int(common.f32_to_bf16(np.asarray([np.float32(q0) * scale]))[0])
    mutated_weight = int(common.f32_to_bf16(np.asarray([np.float32(q1) * scale]))[0])
    original[0] = common.round_f32_bits_to_bf16(
        common.soft_fma_bits(original_weight << 16, 0x3B800000, 0)
    )
    mutated[0] = common.round_f32_bits_to_bf16(
        common.soft_fma_bits(mutated_weight << 16, 0x3B800000, 0)
    )
    evidence = {
        "name": "q_sensitivity_witness",
        "index": index,
        "row": index // common.COLS,
        "column": index % common.COLS,
        "stored": stored,
        "q": q0,
        "step_stored": step,
        "step_q": q1,
        "scale_word": scale_word,
        "weight_words": [original_weight, mutated_weight],
        "output_words": [int(original[0]), int(mutated[0])],
        "changed_words": int(np.count_nonzero(original != mutated)),
        "activation_sha256": sha(activation.tobytes()),
        "original_sha256": sha(original.tobytes()),
        "mutated_sha256": sha(mutated.tobytes()),
    }
    evidence["pass"] = (
        evidence["index"] == 0
        and evidence["q"] == 8
        and evidence["step_q"] == 7
        and evidence["output_words"] == [0x3894, 0x3882]
        and evidence["changed_words"] == 1
        and evidence["activation_sha256"] == EXPECTED_ACTIVATION_SHA
        and evidence["original_sha256"] == EXPECTED_ORIGINAL_SHA
        and evidence["mutated_sha256"] == EXPECTED_MUTATED_SHA
    )
    return evidence


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_r2_exploratory_real_projection",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "prereg_sha256": common.file_digest(PREREG),
            "runner_sha256": common.file_digest(Path(__file__)),
            "prior_result_sha256": "bf10932ad5e67bcb356e49184f57261e1d3453b099b48bba502988eb5743c3c0",
            "diagnostic_sha256": "18b0540a55e2c02fa82db82724994bd3139875884dc3e9ed8c755fa0ee487b54",
        },
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        record, evidence = common.build_record(source)
        safe = common.safe_check(record, input_bytes)
        controls = common.controls(record, input_bytes)[:8] + [sensitivity_witness(record)]
        oracle = common.cpu_oracle(record, input_bytes)
        result.update(
            {
                "record_evidence": evidence,
                "safe_trace": safe["trace"],
                "controls": controls,
                "cpu_output_hex": oracle.tobytes().hex(),
                "cpu_output_sha256": sha(oracle.tobytes()),
                "intel_source_sha256": sha(INTEL_SOURCE.encode()),
                "nvidia_source_sha256": sha(ph0x.CUDA_SOURCE.encode()),
            }
        )
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
    return 0 if error is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
