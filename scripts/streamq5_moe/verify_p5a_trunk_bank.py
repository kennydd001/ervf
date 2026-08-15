from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.build_p5a_trunk_bank import MODEL, PREREG, RESULT, quantize, sha256


R = ROOT / "reports/streamq5_moe"
OUTPUT = R / "p5a_trunk_bank_verification.json"
REPORT = R / "P5A_TRUNK_BANK_VERIFICATION.md"
LOCKED_INDICES = (0, 1, 4, 19, 37, 55, 79, 101, 127, 149, 175, 199, 223, 239, 240)


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P5A verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []
    failures = []
    checked_weights = 0
    for index in LOCKED_INDICES:
        record = result["records"][index]
        with safe_open(MODEL / record["source_shard"], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(record["source_key"])
        expected_codes, expected_scales = quantize(value)
        raw = (ROOT / record["artifact"]).read_bytes()
        observed_codes = np.frombuffer(raw[:record["code_bytes"]], dtype=np.int8).reshape(record["rows"], record["cols"])
        observed_scales = np.frombuffer(raw[record["code_bytes"]:], dtype="<u2").reshape(record["rows"], record["cols"] // 128)
        code_equal = bool(np.array_equal(observed_codes, expected_codes))
        scale_equal = bool(np.array_equal(observed_scales, expected_scales))
        hash_equal = sha256(ROOT / record["artifact"]) == record["artifact_sha256"]
        row = {"record_index":index,"layer":record["layer"],"name":record["name"],"weights":record["weights"],"code_equal":code_equal,"scale_equal":scale_equal,"hash_equal":hash_equal,"pass":code_equal and scale_equal and hash_equal}
        checks.append(row); checked_weights += record["weights"]
        if not row["pass"]: failures.append(row)
        print(json.dumps(row), flush=True)
    arithmetic = {
        "records_241": result["aggregate"]["records"] == 241,
        "weights_exact": result["aggregate"]["weights"] == 1229717504,
        "codes_equal_weights": result["aggregate"]["codes"] == result["aggregate"]["weights"],
        "scales_exact": result["aggregate"]["scales"] == 9607168,
        "bytes_exact": result["aggregate"]["bytes"] == result["aggregate"]["codes"] + 2 * result["aggregate"]["scales"],
        "preregistration_hash": result["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "all_artifact_hashes": all(sha256(ROOT / row["artifact"]) == row["artifact_sha256"] for row in result["records"]),
    }
    passed = not failures and all(arithmetic.values()) and len(checks) == 15
    payload={"kind":"streamq5_moe_p5a_independent_trunk_bank_verification","completed_utc":datetime.now(timezone.utc).isoformat(),"status":"p5a_trunk_bank_verification_pass" if passed else "p5a_trunk_bank_verification_fail","result_sha256":sha256(RESULT),"locked_record_indices":list(LOCKED_INDICES),"records_checked":len(checks),"weights_recomputed":checked_weights,"failures":failures,"checks":checks,"arithmetic":arithmetic,"claim_boundary":"Physical INT8 code/scale bank equivalence only; kernel timing unproven."}
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    REPORT.write_text(f"# P5A trunkbank-verificatie\n\nStatus: **{payload['status']}**. {len(checks)}/15 records en {checked_weights:,} gewichten volledig herberekend.\n",encoding="utf-8")
    print(json.dumps({"status":payload["status"],"records_checked":len(checks),"weights_recomputed":checked_weights,"arithmetic":arithmetic},indent=2))
    if not passed: raise SystemExit(1)


if __name__=="__main__": main()
