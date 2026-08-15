from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe import run_het_next_cap0x_existing_runner_diagnostic as base


RUN = ROOT / "reports/runs/streamq5_moe/het_next_cap0x_r1_import_bootstrap"


def configure() -> None:
    base.RUN = RUN
    base.PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_CAP0X_R1_IMPORT_BOOTSTRAP_REPAIR_PREREGISTRATION_2026-08-13.md"
    base.RESULT = RUN / "cap0x_r1_result.json"
    base.INTEL_RESULT = RUN / "intel_st2_w8.json"
    base.NVIDIA_RESULT = RUN / "nvidia_d7.json"
    base.NVIDIA_REPORT = RUN / "nvidia_d7.md"
    base.__file__ = str(Path(__file__).resolve())
    base.child_command = lambda role: [sys.executable, str(Path(__file__).resolve()), "--role", role]


def main() -> int:
    configure()
    role = "coordinator"
    if "--role" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1]
    if role == "intel":
        return base.run_intel()
    if role == "nvidia":
        return base.run_nvidia()
    original_run = base.subprocess.run
    base.subprocess.run = lambda *args, **kwargs: SimpleNamespace(stdout="poll_disabled_in_cap0x_r1")
    try:
        return base.run_coordinator()
    finally:
        base.subprocess.run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
