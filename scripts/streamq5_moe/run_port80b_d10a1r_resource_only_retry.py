from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe import run_port80b_d10a_next_component_composition as core


REPORTS = ROOT / "reports" / "streamq5_moe"

PREREG = REPORTS / "PORT80B_D10A1R_RESOURCE_ONLY_RETRY_PREREGISTRATION.md"
COMPILE_OUT = REPORTS / "port80b_d10a1r_resource_only_retry_compile.json"
COMPILE_REPORT = REPORTS / "PORT80B_D10A1R_RESOURCE_ONLY_RETRY_COMPILE_REPORT_2026-08-13.md"
COMPONENT_OUT = REPORTS / "port80b_d10a1r_resource_only_retry.json"
COMPONENT_REPORT = REPORTS / "PORT80B_D10A1R_RESOURCE_ONLY_RETRY_REPORT_2026-08-13.md"
ENDURANCE_OUT = REPORTS / "port80b_d10a1r_resource_only_retry_endurance_10k.json"

BASE_PREREG = REPORTS / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_PREREGISTRATION.md"
BASE_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a_next_component_composition.py"
SOURCE_AUDIT = REPORTS / "D10A1_FIRST_TOUCH_INDEPENDENT_SOURCE_AUDIT_2026-08-13.md"
D9_RESOURCE_AUDIT = REPORTS / "port80b_d9_capacity_aware_bank_bridge_independent_verification.json"
OLD_STOP_JSON = REPORTS / "port80b_d10a_next_component_composition_resource_stop.json"
OLD_STOP_REPORT = REPORTS / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_RESOURCE_STOP_REPORT_2026-08-13.md"

EXPECTED = {
    "base_prereg": "ee71a17bf8889e009f8f692aa1dbafddcbd8691b68aa808ab09df0977d607b91",
    "base_runner": "ffde9c13a3d6d19e3e1132369a4eb9a2e98a4e974bbece86ec224e2931f0ecfd",
    "source_audit": "869e56574082e96ce960662dda3dd7e542cd814fb467bf5051831a6efefac081",
    "d9_resource_audit": "629593339d9e39d7ce12d8e85277d6bb9f37ee7316afb426b71529a2c37a6747",
    "old_stop_json": "a884c891eb77e69c0888cb677fdb4d74d51e5732951424e6f0ccedab9bcf9c24",
    "old_stop_report": "2923d75956d86675847c42fe76f3c62794d5af4429f46c984c67576f93230847",
}

EXPECTED_UNIQUE_RECORDS = 14_452
EXPECTED_UNIQUE_MMAP_BYTES = 29_301_719_040
POST_FIRST_TOUCH_RESERVE = 2 * 2**30
START_MARGIN = 1 * 2**30
START_RAM_GATE = EXPECTED_UNIQUE_MMAP_BYTES + POST_FIRST_TOUCH_RESERVE + START_MARGIN
assert START_RAM_GATE == 32_522_944_512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_source_union(routes: dict[str, Any]) -> dict[str, Any]:
    cases = core.selected_cases(routes, core.CORRECTNESS)
    cases.extend(core.selected_cases(routes, core.VALIDATION, 32))
    records = {
        (layer, int(expert))
        for case in cases
        for layer, row in enumerate(case["route"])
        for expert in row
    }
    routed_records = len(records)
    records.update((layer, 512) for layer in range(core.LAYERS))
    records.update((0, expert) for expert in range(core.TOP_K))

    first = core.selected_cases(routes, core.CORRECTNESS, 1)[0]["route"]
    flat = first.reshape(-1)
    injection_index = next(index for index, expert in enumerate(flat) if int(expert) < core.PREFIX - 1)
    injection_layer = injection_index // core.TOP_K
    intended_expert = int(flat[injection_index])
    records.add((injection_layer, intended_expert + 1))
    records.add(((injection_layer + 1) % core.LAYERS, intended_expert))

    return {
        "method": "exact set union over the frozen 40 correctness and first 32 validation routes, 48 shared records, ten layer-0 references, and two negative-control sources",
        "routed_union_records": routed_records,
        "complete_unique_records": len(records),
        "expert_bytes": core.EXPERT_BYTES,
        "unique_mmap_bytes": len(records) * core.EXPERT_BYTES,
        "post_first_touch_reserve_bytes": POST_FIRST_TOUCH_RESERVE,
        "explicit_margin_bytes": START_MARGIN,
        "start_ram_gate_bytes": len(records) * core.EXPERT_BYTES + POST_FIRST_TOUCH_RESERVE + START_MARGIN,
    }


base_audit = core.audit
base_component_phase = core.component_phase


def retry_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    evidence, routes, route_hashes = base_audit()
    union = exact_source_union(routes)
    locks = {
        "base_prereg": sha256(BASE_PREREG),
        "base_runner": sha256(BASE_RUNNER),
        "source_audit": sha256(SOURCE_AUDIT),
        "d9_resource_audit": sha256(D9_RESOURCE_AUDIT),
        "old_stop_json": sha256(OLD_STOP_JSON),
        "old_stop_report": sha256(OLD_STOP_REPORT),
    }
    retry_checks = {
        "all_resource_and_immutability_locks": locks == EXPECTED,
        "routed_union_records_exact": union["routed_union_records"] == 14_404,
        "complete_unique_records_exact": union["complete_unique_records"] == EXPECTED_UNIQUE_RECORDS,
        "unique_mmap_bytes_exact": union["unique_mmap_bytes"] == EXPECTED_UNIQUE_MMAP_BYTES,
        "start_formula_exact": union["start_ram_gate_bytes"] == START_RAM_GATE,
        "only_start_gate_changed": (
            core.MIN_RAM_BEFORE == START_RAM_GATE
            and core.MIN_RAM_AFTER_TOUCH == 2 * 2**30
            and core.EMERGENCY_RAM == int(1.5 * 2**30)
            and core.VRAM_RESERVE == 512 * 2**20
        ),
    }
    evidence["resource_only_retry"] = {
        "independent_resource_audits": [
            {"path": str(SOURCE_AUDIT), "sha256": locks["source_audit"]},
            {"path": str(D9_RESOURCE_AUDIT), "sha256": locks["d9_resource_audit"]},
        ],
        "base_and_prior_stop_locks": locks,
        "source_union": union,
        "checks": retry_checks,
        "pass": all(retry_checks.values()),
        "delta": {
            "old_start_gate_bytes": 50 * 2**30,
            "new_start_gate_bytes": START_RAM_GATE,
            "post_registration_gate_bytes": core.MIN_RAM_AFTER_TOUCH,
            "post_first_touch_gate_bytes": core.MIN_RAM_AFTER_TOUCH,
            "emergency_gate_bytes": core.EMERGENCY_RAM,
        },
    }
    evidence["checks"]["resource_only_retry_exact_and_bound"] = all(retry_checks.values())
    evidence["pass"] = all(evidence["checks"].values())
    return evidence, routes, route_hashes


def component_phase() -> None:
    available = int(psutil.virtual_memory().available)
    if available < START_RAM_GATE:
        raise RuntimeError(
            f"hard stop: available RAM {available} < D10A1-R exact start gate {START_RAM_GATE}"
        )
    base_component_phase()


# Rebind the unmodified D10A1 executor to retry-only paths and the one authorized
# resource delta. Its CUDA, correctness, validation, post-touch, emergency and
# cleanup logic remains the hash-locked base implementation.
core.__file__ = str(Path(__file__).resolve())
core.PREREG = PREREG
core.COMPILE_OUT = COMPILE_OUT
core.COMPILE_REPORT = COMPILE_REPORT
core.COMPONENT_OUT = COMPONENT_OUT
core.REPORT = COMPONENT_REPORT
core.ENDURANCE_OUT = ENDURANCE_OUT
core.MIN_RAM_BEFORE = START_RAM_GATE
core.audit = retry_audit
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
