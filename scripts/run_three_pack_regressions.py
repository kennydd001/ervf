from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

from moe_lab.reporting import ROOT


OUT = ROOT / "reports/three_pack_regression_tests.json"


if __name__ == "__main__":
    if OUT.exists():
        raise FileExistsError("refusing to overwrite regression artifact")
    command = [sys.executable, "-m", "pytest", "-q"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    payload = {
        "kind": "three_pack_regression_tests",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "passed": process.returncode == 0 and "153 passed" in process.stdout,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    raise SystemExit(process.returncode)
