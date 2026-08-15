from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe import run_port80b_d10a_next_component_composition as core


REPORTS = ROOT / "reports" / "streamq5_moe"
PREREG = REPORTS / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_PREREGISTRATION.md"
COMPILE_OUT = REPORTS / "port80b_d10a1r_conservative_resource_retry_compile.json"
COMPILE_REPORT = REPORTS / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_COMPILE_REPORT_2026-08-13.md"
COMPONENT_OUT = REPORTS / "port80b_d10a1r_conservative_resource_retry.json"
COMPONENT_REPORT = REPORTS / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_REPORT_2026-08-13.md"
ENDURANCE_OUT = REPORTS / "port80b_d10a1r_conservative_resource_retry_endurance_10k.json"

BASE_PREREG = REPORTS / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_PREREGISTRATION.md"
BASE_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a_next_component_composition.py"
SOURCE_AUDIT = REPORTS / "D10A1_FIRST_TOUCH_INDEPENDENT_SOURCE_AUDIT_2026-08-13.md"
BUDGET_AUDIT = REPORTS / "port80b_d10a1r_resource_budget_audit.json"
OLD_STOP_JSON = REPORTS / "port80b_d10a_next_component_composition_resource_stop.json"
OLD_STOP_REPORT = REPORTS / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_RESOURCE_STOP_REPORT_2026-08-13.md"

EXPECTED_LOCKS = {
    "base_prereg": "ee71a17bf8889e009f8f692aa1dbafddcbd8691b68aa808ab09df0977d607b91",
    "base_runner": "ffde9c13a3d6d19e3e1132369a4eb9a2e98a4e974bbece86ec224e2931f0ecfd",
    "source_audit": "869e56574082e96ce960662dda3dd7e542cd814fb467bf5051831a6efefac081",
    "budget_audit": "8a79cc68afa2e1e43373b9990b8cbadc9cf9b51ac811ddf2a142afea6922789f",
    "old_stop_json": "a884c891eb77e69c0888cb677fdb4d74d51e5732951424e6f0ccedab9bcf9c24",
    "old_stop_report": "2923d75956d86675847c42fe76f3c62794d5af4429f46c984c67576f93230847",
}

EXPECTED_ROUTE_UNION_RECORDS = 14_452
EXPECTED_ROUTE_UNION_BYTES = 29_301_719_040
REGISTERED_PREFIX_RECORDS = core.LAYERS * core.PREFIX
EXPECTED_EXPLICIT_OUTSIDE_PREFIX_RECORDS = 428
EXPECTED_CONSERVATIVE_RECORDS = 24_380
EXPECTED_CONSERVATIVE_BYTES = 49_430_937_600
HOST_ALLOWANCE = 1 * 2**30
POST_TOUCH_RESERVE = 2 * 2**30
START_RAM_GATE = EXPECTED_CONSERVATIVE_BYTES + HOST_ALLOWANCE + POST_TOUCH_RESERVE
assert START_RAM_GATE == 52_652_163_072


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_touched_records(routes: dict[str, Any]) -> set[tuple[int, int]]:
    cases = core.selected_cases(routes, core.CORRECTNESS)
    cases.extend(core.selected_cases(routes, core.VALIDATION, 32))
    records = {
        (layer, int(expert))
        for case in cases
        for layer, row in enumerate(case["route"])
        for expert in row
    }
    records.update((layer, 512) for layer in range(core.LAYERS))
    records.update((0, expert) for expert in range(core.TOP_K))
    first = core.selected_cases(routes, core.CORRECTNESS, 1)[0]["route"]
    flat = first.reshape(-1)
    index = next(i for i, expert in enumerate(flat) if int(expert) < core.PREFIX - 1)
    layer = index // core.TOP_K
    expert = int(flat[index])
    records.add((layer, expert + 1))
    records.add(((layer + 1) % core.LAYERS, expert))
    return records


base_audit = core.audit
base_component_phase = core.component_phase


def conservative_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    evidence, routes, route_hashes = base_audit()
    touched = exact_touched_records(routes)
    outside = {(layer, expert) for layer, expert in touched if expert >= core.PREFIX}
    conservative_records = REGISTERED_PREFIX_RECORDS + len(outside)
    locks = {
        "base_prereg": sha256(BASE_PREREG),
        "base_runner": sha256(BASE_RUNNER),
        "source_audit": sha256(SOURCE_AUDIT),
        "budget_audit": sha256(BUDGET_AUDIT),
        "old_stop_json": sha256(OLD_STOP_JSON),
        "old_stop_report": sha256(OLD_STOP_REPORT),
    }
    budget = json.loads(BUDGET_AUDIT.read_text(encoding="utf-8"))
    calculated = {
        "route_union_records": len(touched),
        "route_union_bytes": len(touched) * core.EXPERT_BYTES,
        "registered_prefix_records": REGISTERED_PREFIX_RECORDS,
        "explicit_outside_prefix_records": len(outside),
        "conservative_union_records": conservative_records,
        "conservative_union_bytes": conservative_records * core.EXPERT_BYTES,
        "host_allowance_bytes": HOST_ALLOWANCE,
        "post_touch_reserve_bytes": POST_TOUCH_RESERVE,
        "start_ram_gate_bytes": conservative_records * core.EXPERT_BYTES + HOST_ALLOWANCE + POST_TOUCH_RESERVE,
        "start_ram_gate_gib": START_RAM_GATE / 2**30,
    }
    checks = {
        "all_immutable_locks": locks == EXPECTED_LOCKS,
        "both_independent_audits_pass": (
            budget.get("pass") is True and SOURCE_AUDIT.is_file()
        ),
        "route_union_records_exact": len(touched) == EXPECTED_ROUTE_UNION_RECORDS,
        "route_union_bytes_exact": len(touched) * core.EXPERT_BYTES == EXPECTED_ROUTE_UNION_BYTES,
        "registered_prefix_records_exact": REGISTERED_PREFIX_RECORDS == 23_952,
        "explicit_outside_prefix_records_exact": len(outside) == EXPECTED_EXPLICIT_OUTSIDE_PREFIX_RECORDS,
        "conservative_union_records_exact": conservative_records == EXPECTED_CONSERVATIVE_RECORDS,
        "conservative_union_bytes_exact": conservative_records * core.EXPERT_BYTES == EXPECTED_CONSERVATIVE_BYTES,
        "budget_audit_formula_exact": (
            budget["registration_residency_caveat"]["registered_prefix_plus_explicit_outside_prefix"]["bytes"]
            == EXPECTED_CONSERVATIVE_BYTES
            and budget["budget"]["safe_minimum_starting_psutil_available_bytes"] == START_RAM_GATE
        ),
        "start_formula_exact": calculated["start_ram_gate_bytes"] == START_RAM_GATE,
        "only_start_gate_changed": (
            core.MIN_RAM_BEFORE == START_RAM_GATE
            and core.MIN_RAM_AFTER_TOUCH == 2 * 2**30
            and core.EMERGENCY_RAM == int(1.5 * 2**30)
            and core.VRAM_RESERVE == 512 * 2**20
        ),
    }
    evidence["resource_only_retry"] = {
        "policy": "conservative full possible registered-prefix plus explicit outside-prefix residency",
        "independent_resource_audits": [
            {"path": str(SOURCE_AUDIT), "sha256": locks["source_audit"]},
            {"path": str(BUDGET_AUDIT), "sha256": locks["budget_audit"]},
        ],
        "base_and_prior_stop_locks": locks,
        "calculated": calculated,
        "checks": checks,
        "pass": all(checks.values()),
        "rejected_less_conservative_gate_bytes": 32_522_944_512,
        "delta": {
            "old_d10a_start_gate_bytes": 50 * 2**30,
            "new_d10a1r_start_gate_bytes": START_RAM_GATE,
            "post_registration_gate_bytes": core.MIN_RAM_AFTER_TOUCH,
            "post_first_touch_gate_bytes": core.MIN_RAM_AFTER_TOUCH,
            "emergency_gate_bytes": core.EMERGENCY_RAM,
        },
    }
    evidence["checks"]["resource_only_retry_conservative_exact_and_bound"] = all(checks.values())
    evidence["pass"] = all(evidence["checks"].values())
    return evidence, routes, route_hashes


def component_phase() -> None:
    available = int(psutil.virtual_memory().available)
    if available < START_RAM_GATE:
        raise RuntimeError(
            f"hard stop: available RAM {available} < D10A1-R conservative start gate {START_RAM_GATE}"
        )
    base_component_phase()


core.__file__ = str(Path(__file__).resolve())
core.PREREG = PREREG
core.COMPILE_OUT = COMPILE_OUT
core.COMPILE_REPORT = COMPILE_REPORT
core.COMPONENT_OUT = COMPONENT_OUT
core.REPORT = COMPONENT_REPORT
core.ENDURANCE_OUT = ENDURANCE_OUT
core.MIN_RAM_BEFORE = START_RAM_GATE
core.audit = conservative_audit
core.component_phase = component_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "component", "endurance"), required=True)
    parser.add_argument("--acknowledge-endurance")
    args = parser.parse_args()
    if args.phase == "compile":
        core.compile_phase()
    elif args.phase == "component":
        component_phase()
    else:
        core.endurance_phase(args.acknowledge_endurance)


if __name__ == "__main__":
    main()
