from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
PREREG = REPORTS / "P8D_Q5_CODE_AUDIT_PREREGISTRATION.md"
SOURCE = Path(__file__).with_name("p8d_q5_code_audit.cpp")
OUTPUT = REPORTS / "p8d_q5_code_audit.json"
REPORT = REPORTS / "P8D_Q5_CODE_AUDIT.md"
BANK = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive[0].lower()
    suffix = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def main() -> None:
    compile_command = (
        f"g++ -O3 -march=native -fopenmp -std=c++17 '{wsl(SOURCE)}' "
        "-o /tmp/p8d_q5_code_audit"
    )
    subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", compile_command], check=True)
    command = (
        "OMP_NUM_THREADS=16 /tmp/p8d_q5_code_audit "
        f"'{wsl(BANK)}'"
    )
    process = subprocess.Popen(
        ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stderr is not None
    for line in process.stderr:
        print(line.rstrip(), flush=True)
    stdout, _ = process.communicate()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    result = json.loads(stdout)
    result.update({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "cpp_source_sha256": sha256(SOURCE),
        "bank_manifest_sha256": sha256(REPORTS / "p1d_physical_bank_result.json"),
        "all_count_gates_pass": bool(
            result["codes"] == 28_991_029_248
            and result["records"] == 18_432
            and all(value == 603_979_776 for value in result["layer_codes"])
            and all(sum(values) == 9_663_676_416 for values in result["projection_histograms"].values())
        ),
    })
    overflow = result["overflow_fraction"]
    ratio = result["conservative_int4_plus_index_value_ratio_to_q5"]
    if overflow < 0.10:
        decision = "candidate_for_physical_layout_test"
    elif overflow <= 0.20 and ratio <= 0.85:
        decision = "candidate_due_to_at_least_15_percent_conservative_saving"
    else:
        decision = "naive_int4_core_sparse_overflow_falsified"
    result["decision"] = decision
    result["overall_pass"] = bool(result["all_count_gates_pass"] and decision.startswith("candidate"))
    result["claim_boundary"] = (
        "Exact code-distribution and conservative storage result for the full local Q5 bank; "
        "it does not evaluate entropy-code decode speed."
    )
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    REPORT.write_text(
        "# P8D exact Q5-codeaudit\n\n"
        f"- Records: `{result['records']:,}`\n"
        f"- Codes: `{result['codes']:,}`\n"
        f"- Entropie: `{result['entropy_bits_per_code']:.6f}` bit/code\n"
        f"- `|code| > 7`: `{result['overflow_fraction']:.6%}`\n"
        f"- Conservatieve INT4+index+waarde/Q5-ratio: `{ratio:.6f}`\n"
        f"- Besluit: **{decision}**\n\n"
        "De eenvoudige verliesloze overflowlijst is hiermee gesloten. Entropiecodering is een "
        "afzonderlijke hypothese omdat zij sequentiële decode- en indexkosten introduceert.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in (
        "codes", "records", "entropy_bits_per_code", "overflow_fraction",
        "conservative_int4_plus_index_value_ratio_to_q5", "all_count_gates_pass",
        "decision", "overall_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
