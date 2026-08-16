"""Dependency-aware orchestrator for PRO-MAX V2."""
from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import LOGS, ensure_dirs, load_json, result_path, slug, write_text


def run_step(name: str, script: str, *args: str) -> int:
    ensure_dirs()
    log = LOGS / f"{slug()}__{name}.log"
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {name} ===\n{' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=HERE.parent.parent, text=True, capture_output=True)
    text = (f"COMMAND: {' '.join(cmd)}\nRETURN CODE: {p.returncode}\n\n"
            f"STDOUT\n------\n{p.stdout}\n\nSTDERR\n------\n{p.stderr}\n")
    write_text(log, text, archive=False)
    if p.stdout: print(p.stdout, end="")
    if p.stderr: print(p.stderr, file=sys.stderr, end="")
    print(f"log: {log}")
    return int(p.returncode)


def install() -> int:
    scripts = sorted(HERE.glob("*.py"))
    failures = []
    for path in scripts:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    try:
        import yaml
        yaml.safe_load((HERE / "POST_V6_REGISTRY.yaml").read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"registry: {exc}")
    print(json.dumps({"python_files": len(scripts), "failures": failures,
                      "status": "pass" if not failures else "fail"}, indent=2))
    return 0 if not failures else 2


def result_adopt(name: str) -> bool:
    path = result_path(name)
    if not path.exists(): return False
    try: return bool(load_json(path).get("adopt"))
    except Exception: return False


def experimental(mode: str, include_architecture: bool) -> int:
    rc = 0
    rc |= run_step("PV2_00", "provenance_budget.py")
    rc |= run_step("PV2_10", "addnorm_v7.py", "--mode", mode)
    rc |= run_step("PV2_11", "qkv_v8.py", "--mode", mode)
    rc |= run_step("PV2_12", "lmhead_argmax_v9.py", "--mode", mode)
    rc |= run_step("PV2_13", "finale_v10.py", "--mode", mode)
    if include_architecture:
        rc |= run_step("PV2_20", "child_epoch_v11.py", "--mode", mode)
        rc |= run_step("PV2_21", "capabilities_v2.py")
    rc |= run_step("PV2_VERIFY", "verify.py")
    rc |= run_step("PV2_REPORT", "report.py")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("install", "smoke", "full", "architecture", "overnight", "verify", "report"))
    args = ap.parse_args()
    if args.mode == "install": return install()
    if args.mode == "smoke": return experimental("smoke", True)
    if args.mode == "full": return experimental("full", False)
    if args.mode == "architecture":
        rc = run_step("PV2_20", "child_epoch_v11.py", "--mode", "full")
        rc |= run_step("PV2_21", "capabilities_v2.py")
        rc |= run_step("PV2_VERIFY", "verify.py")
        rc |= run_step("PV2_REPORT", "report.py")
        return rc
    if args.mode == "overnight": return experimental("full", True)
    if args.mode == "verify": return run_step("PV2_VERIFY", "verify.py")
    return run_step("PV2_REPORT", "report.py")

if __name__ == "__main__":
    raise SystemExit(main())
