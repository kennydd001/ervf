from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.build_p6a_exact_runtime_bank import quantize_exact_p0c, sha256


MODEL = ROOT / "models/qwen3-30b-a3b-base"
R = ROOT / "reports/streamq5_moe"
RESULT = R / "p6a_exact_runtime_bank_result.json"
OUTPUT = R / "p6a_exact_runtime_bank_verification.json"
REPORT = R / "P6A_EXACT_RUNTIME_BANK_VERIFICATION.md"
P5A_RESULT = R / "p5a_trunk_bank_result.json"

SELECTED = [
    (0, "q"), (0, "k"), (0, "router"), (3, "router"), (7, "v"),
    (11, "q"), (15, "router"), (20, "k"), (25, "v"), (31, "o"),
    (32, "k"), (37, "q"), (42, "router"), (47, "o"), (48, "head"),
    (49, "embed"),
]


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P6A bank verification")
    bank = json.loads(RESULT.read_text(encoding="utf-8"))
    p5a = json.loads(P5A_RESULT.read_text(encoding="utf-8"))
    by_key = {(r["layer"], r["name"]): r for r in bank["records"]}
    p5_by_key = {(r["layer"], r["name"]): r for r in p5a["records"]}
    checks = []
    discrepancy = []
    selected_by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for key in SELECTED:
        selected_by_shard[by_key[key]["source_shard"]].append(key)
    for shard, keys in selected_by_shard.items():
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                record = by_key[key]
                value = handle.get_tensor(record["source_key"])
                codes, scales = quantize_exact_p0c(value)
                path = ROOT / record["artifact"]
                raw = path.read_bytes()
                observed_codes = np.frombuffer(raw, dtype=np.int8, count=record["weights"])
                observed_scales = np.frombuffer(raw, dtype="<u2", count=record["groups"], offset=record["code_bytes"])
                code_equal = bool(np.array_equal(observed_codes, codes.reshape(-1)))
                scale_equal = bool(np.array_equal(observed_scales, scales.reshape(-1)))
                check = {
                    "layer": key[0], "name": key[1], "weights": record["weights"],
                    "artifact_hash_equal": sha256(path) == record["artifact_sha256"],
                    "codes_equal_source": code_equal, "scales_equal_source": scale_equal,
                }
                check["pass"] = check["artifact_hash_equal"] and code_equal and scale_equal
                checks.append(check)
                if key in p5_by_key:
                    old = p5_by_key[key]
                    old_raw = (ROOT / old["artifact"]).read_bytes()
                    old_codes = np.frombuffer(old_raw, dtype=np.int8, count=old["weights"])
                    different = int(np.count_nonzero(old_codes != observed_codes))
                    discrepancy.append({
                        "layer": key[0], "name": key[1], "codes": record["weights"],
                        "different_codes": different,
                        "different_fraction": different / record["weights"],
                    })

    norm_path = ROOT / bank["norm_bank"]["artifact"]
    norm_raw = norm_path.read_bytes()
    norm_checks = []
    norms_by_shard: dict[str, list[dict]] = defaultdict(list)
    for record in bank["norm_bank"]["records"]:
        norms_by_shard[record["source_shard"]].append(record)
    for shard, records in norms_by_shard.items():
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for record in records:
                expected = handle.get_tensor(record["source_key"]).to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(order="C")
                observed = norm_raw[record["offset"]:record["offset"] + record["bytes"]]
                norm_checks.append({
                    "layer": record["layer"], "name": record["name"],
                    "payload_equal_source": observed == expected,
                    "payload_hash_equal": hashlib.sha256(observed).hexdigest() == record["payload_sha256"],
                })

    all_hashes = all(sha256(ROOT / r["artifact"]) == r["artifact_sha256"] for r in bank["records"])
    aggregate = bank["aggregate"]
    arithmetic = {
        "records_242": len(bank["records"]) == aggregate["records"] == 242,
        "device_records_241": aggregate["device_records"] == 241,
        "embedding_records_1": aggregate["host_records"] == 1,
        "device_bytes_exact": aggregate["device_bytes"] == 1_248_931_840,
        "embedding_bytes_exact": aggregate["host_embedding_bytes"] == 316_026_880,
        "total_bytes_exact": aggregate["bytes"] == sum(r["bytes"] for r in bank["records"]),
        "weights_equal_codes": aggregate["weights"] == aggregate["codes"],
        "norm_records_193": len(norm_checks) == 193,
        "norm_bytes_exact": len(norm_raw) == bank["norm_bank"]["bytes"] == 421_888,
        "all_artifact_hashes": all_hashes,
        "norm_artifact_hash": sha256(norm_path) == bank["norm_bank"]["artifact_sha256"],
        "preregistration_hash": bank["inputs"]["preregistration_sha256"] == sha256(R / "P6A_END_TO_END_DECODE_PREREGISTRATION.md"),
    }
    passed = all(arithmetic.values()) and len(checks) == 16 and all(c["pass"] for c in checks) and all(c["payload_equal_source"] and c["payload_hash_equal"] for c in norm_checks)
    payload = {
        "kind": "streamq5_moe_p6a_exact_runtime_bank_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p6a_exact_runtime_bank_verification_pass" if passed else "p6a_exact_runtime_bank_verification_fail",
        "inputs": {"bank_result_sha256": sha256(RESULT), "p5a_bank_result_sha256": sha256(P5A_RESULT)},
        "arithmetic": arithmetic,
        "selected_record_checks": checks,
        "norm_checks": norm_checks,
        "p5a_semantic_discrepancy_sample": discrepancy,
        "p5a_sample_different_codes": sum(r["different_codes"] for r in discrepancy),
        "p5a_sample_codes": sum(r["codes"] for r in discrepancy),
        "claim_boundary": "Independent bank/source equivalence and P5A code-selection discrepancy only; runtime remains unopened.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fraction = payload["p5a_sample_different_codes"] / max(payload["p5a_sample_codes"], 1)
    REPORT.write_text(
        "# P6A exacte runtimebank — onafhankelijke verificatie\n\n"
        f"Status: **{payload['status']}**. Alle {len(checks)} geselecteerde Q8-records "
        f"en {len(norm_checks)} normrecords zijn opnieuw uit het checkpoint afgeleid. "
        f"In de vergelijkbare P5A-sample verschilde {fraction:.6%} van de codes door de "
        "andere schaalvolgorde.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "checks": len(checks), "norm_checks": len(norm_checks), "p5a_sample_different_fraction": fraction, "arithmetic": arithmetic}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
