from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("p11a_stream.cpp")
OUTPUT = ROOT / "reports/streamq5_moe/p11a_cpu_stream.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wsl(path: Path) -> str:
    resolved = path.resolve(); return f"/mnt/{resolved.drive[0].lower()}{resolved.as_posix().split(':', 1)[1]}"


command = f"g++ -O3 -march=native -fopenmp -std=c++20 '{wsl(SOURCE)}' -o /tmp/p11a_stream"
subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command], check=True)
measured = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "/tmp/p11a_stream"], check=True, capture_output=True, text=True)
result = json.loads(measured.stdout)
result.update({
    "kind": "streamq5_moe_p11a_cpu_stream", "completed_utc": datetime.now(timezone.utc).isoformat(),
    "script_sha256": sha256(Path(__file__)), "cpp_source_sha256": sha256(SOURCE),
    "cpu": "Intel Core Ultra 9 285H", "logical_and_physical_cpus_visible": 16,
    "numa_nodes_visible": 1, "explicit_hugepages_total": 0,
    "claim_boundary": "WSL2 STREAM-like diagnostic; Windows pinned-memory and native kernels can differ.",
})
OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
