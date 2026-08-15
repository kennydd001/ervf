"""Emit a source-to-claim hash table from the frozen protected manifest.

Every historical claim quoted by the Nemotron line must be traceable to an exact
protected file and its SHA-256 as recorded at the moment this research line
started.  Reading the hashes out of the baseline manifest rather than rehashing
guarantees the table describes the state that was frozen, not a later state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "reports" / "lightningstream_nemotron" / "PROTECTED_80B_MANIFEST_BEFORE.json"

CITED = [
    "README.md",
    "docs/RESEARCH_LOG.md",
    "docs/PRIOR_ART.md",
    "docs/CRAFT_MOE_PRIOR_ART.md",
    "docs/RSIV_MOE_RESEARCH_LOG.md",
    "docs/CORETAIL_MOE_RESEARCH_LOG.md",
    "docs/OFFLOAD_ROOFLINE_RESEARCH_LOG.md",
    "reports/BASELINE_2026-08-09.md",
    "reports/RESULTS_2026-08-09.md",
    "reports/EUREKA_VERDICT_2026-08-10.md",
    "reports/MASS_BUDGET_EUREKA_2026-08-10.md",
    "reports/PREREGISTERED_MASS_BUDGET_CONFIRMATION_2026-08-10.md",
    "reports/FINAL_EUREKA_VERDICT_2026-08-12.md",
    "reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md",
    "reports/streamq5_moe/FINAL_VERDICT.md",
    "reports/streamq5_moe/EXPERIMENT_REGISTRY.yaml",
    "reports/streamq5_moe/ALL_IDEAS_FINAL_REPORT_2026-08-12.md",
    "reports/streamq5_moe/ALL_IDEAS_CLOSURE_REGISTRY_2026-08-12.yaml",
    "reports/streamq5_moe/BREAKTHROUGH_RESEARCH_FINAL_REPORT_2026-08-12.md",
    "reports/streamq5_moe/BREAKTHROUGH_PHASE_REGISTRY_2026-08-12.yaml",
    "reports/streamq5_moe/DATAPLANE_ONTLEDING_VERDICT_2026-08-12.md",
    "reports/streamq5_moe/PORT80B_D10_ARCHITECTURE_AUDIT_AND_DESIGN_2026-08-13.md",
    "reports/streamq5_moe/PORT80B_T0Q5_S0R5_C1R2A_COMBINED_REPORT_2026-08-13.md",
    "reports/streamq5_moe/PH1_INTEL_R8A5_FINAL_COMPONENT_REPORT_2026-08-14.md",
    "reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8v1r1a_independent_verification.json",
    "reports/streamq5_moe/HET_NEXT_L0_PH1_NVIDIA_FULL_EXPERT_N5_IMPLEMENTATION_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "reports/streamq5_moe/NEMOTRON_N0_METADATA_GATE_REPORT_2026-08-12.md",
    "reports/streamq5_moe/nemotron_n0_metadata_gate.json",
    "reports/streamq5_moe/NEMOTRON_N1_HEADER_INVENTORY_PREREGISTRATION.md",
    "reports/streamq5_moe/NEMOTRON_N1_HEADER_INVENTORY_REPORT_2026-08-12.md",
    "reports/streamq5_moe/nemotron_n1_header_inventory.json",
    "reports/streamq5_moe/P7_ERVF_FINAL_REPORT_2026-08-12.md",
    "reports/streamq5_moe/p7_ervf_independent_verification.json",
    "info/ERVF_EUREKA_2026-08-12.md",
    "info/RAND_VAN_WAT_2026-08-12.md",
    "info/NA_P13C_VIER_HEFBOMEN_2026-08-12.md",
    "info/KERNEL_INVERSIE_2026-08-12.md",
    "info/RICHTINGEN_DOORGEREKEND_2026-08-12.md",
    "info/BITBREEDTE_ANALYSE_2026-08-12.md",
    "info/PORT80B_DIRECTPATH_PACK_2026-08-12/PORT80B_DIRECTPATH_SMOKING_GUN_REPORT_2026-08-12.md",
    "info/PORT80B_DIRECTPATH_PACK_2026-08-12/PORT80B_DIRECTPATH_CALCULATIONS_2026-08-12.json",
    "research_docs_nuttige_info_2026-08-14_111334_compleet.zip",
]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    rows = []
    missing = []
    for rel in CITED:
        entry = files.get(rel)
        if entry is None:
            missing.append(rel)
            continue
        if entry["tier"] == "full":
            digest = entry["sha256"]
        else:
            digest = f"{entry.get('sha256_head', '?')} (head) / {entry.get('sha256_tail', '?')} (tail)"
        rows.append((rel, entry["bytes"], entry["tier"], digest))

    print(f"| # | protected path | bytes | tier | SHA-256 |")
    print(f"|---:|---|---:|---|---|")
    for index, (rel, size, tier, digest) in enumerate(rows, start=1):
        print(f"| {index} | `{rel}` | {size:,} | {tier} | `{digest}` |")

    if missing:
        print("\nMISSING FROM MANIFEST:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
    print(f"\nbaseline root_digest = {manifest['root_digest']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
