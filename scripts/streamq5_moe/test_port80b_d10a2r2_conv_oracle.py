from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe import run_port80b_d10a2r2_gdn36_oracle_repair as runner


OUT = ROOT / "reports" / "streamq5_moe" / "port80b_d10a2r2_conv_oracle_unit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable unit result: {OUT}")
    audit = runner.conv_oracle_unit_audit()
    result = {
        "kind": "port80b_d10a2r2_conv_oracle_cpu_unit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pass": audit["pass"],
        "inputs": {
            "runner_sha256": sha256(Path(runner.__file__)),
            "preregistration_sha256": sha256(runner.PREREG),
            "unit_test_sha256": sha256(Path(__file__)),
        },
        "audit": audit,
        "physical_actions": {
            "cuda_initialized": False,
            "nvrtc_compile": False,
            "host_registration": False,
            "large_device_allocation": False,
            "kernel_launch": False,
            "bank_scan": False,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
