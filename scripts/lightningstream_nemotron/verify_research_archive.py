"""Verify the combined research archive: integrity plus key-document presence."""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

KEY = [
    "README.md",
    "reports/streamq5_moe/FINAL_VERDICT.md",
    "reports/streamq5_moe/PORT80B_D10_ARCHITECTURE_AUDIT_AND_DESIGN_2026-08-13.md",
    "reports/streamq5_moe/PH1_INTEL_R8A5_FINAL_COMPONENT_REPORT_2026-08-14.md",
    "reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md",
    "reports/FINAL_EUREKA_VERDICT_2026-08-12.md",
    "reports/MASS_BUDGET_EUREKA_2026-08-10.md",
    "docs/RESEARCH_LOG.md",
    "docs/PRIOR_ART.md",
    "docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md",
    "reports/lightningstream_nemotron/EXPERIMENT_REGISTRY.yaml",
    "reports/lightningstream_nemotron/LIGHTNINGSTREAM_RESEARCH_HANDOFF.md",
    "reports/lightningstream_nemotron/N7C_VECTORISED_LOADS_REPORT_2026-08-14.md",
    "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py",
    "src/moe_lab/lightningstream_nemotron/runtime.py",
    "src/moe_lab/lightningstream_nemotron/gpu_kernels.py",
    "ARCHIVE_MANIFEST.json",
]


def main() -> int:
    archives = sorted(OUT_DIR.glob("ALL_RESEARCH_LINES_*.zip"))
    if not archives:
        print("no archive found")
        return 3
    path = archives[-1]
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        names = set(z.namelist())

    print(f"archive   : {path.name}")
    print(f"size      : {path.stat().st_size:,} B "
          f"({path.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"integrity : {'OK' if bad is None else 'CORRUPT at ' + bad}")
    print(f"entries   : {len(names):,}")

    missing = [k for k in KEY if k not in names]
    print(f"key docs  : {len(KEY) - len(missing)}/{len(KEY)}")
    for m in missing:
        print(f"  MISSING: {m}")

    for label, needle in (("codex NC0-NC13", "NVIDIA_NC"),
                          ("het-next", "HET_NEXT"),
                          ("port80b", "PORT80B"),
                          ("nemotron line", "lightningstream_nemotron"),
                          ("registries", "EXPERIMENT_REGISTRY")):
        print(f"  {label:<16}: {sum(1 for n in names if needle in n):>5} files")

    return 0 if (bad is None and not missing) else 3


if __name__ == "__main__":
    raise SystemExit(main())
