"""Dependency-aware orchestrator for the additive PRO research pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import LOGS, PRO, load_json, result_path, timestamp_slug, write_text_atomic


def run_step(name: str, args: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{timestamp_slug()}__{name}.log"
    command = [sys.executable, *args]
    print(f"\n=== {name} ===\n{' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=PRO.parent, text=True, capture_output=True)
    text = (
        f"COMMAND: {' '.join(command)}\nRETURN CODE: {proc.returncode}\n\n"
        f"STDOUT\n------\n{proc.stdout}\n\nSTDERR\n------\n{proc.stderr}\n"
    )
    write_text_atomic(log_path, text, archive=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    print(f"log: {log_path}")
    return int(proc.returncode)


def result_status(name: str) -> str | None:
    path = result_path(name)
    if not path.exists():
        return None
    try:
        return str(load_json(path).get("status"))
    except Exception:
        return "unreadable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=("smoke", "graph", "dense", "epoch", "full", "verify", "report"),
        nargs="?",
        default="full",
    )
    args = ap.parse_args()

    rc = 0
    if args.mode == "smoke":
        rc |= run_step("graph_smoke", [str(PRO / "graph_e1f22.py"), "--mode", "smoke", "--skip-control"])
        rc |= run_step("dense_smoke", [str(PRO / "ervf_dense.py"), "--mode", "smoke"])
        if result_status("PRO_G0_E1F22_GRAPH_AB.json") in {"pass", "smoke_pass"}:
            rc |= run_step("epoch_smoke", [str(PRO / "epoch_graph.py"), "--mode", "smoke"])
    elif args.mode == "graph":
        rc |= run_step("graph_full", [str(PRO / "graph_e1f22.py"), "--mode", "full"])
    elif args.mode == "dense":
        rc |= run_step("dense_full", [str(PRO / "ervf_dense.py"), "--mode", "full"])
    elif args.mode == "epoch":
        rc |= run_step("epoch_full", [str(PRO / "epoch_graph.py"), "--mode", "full"])
    elif args.mode == "full":
        rc |= run_step("graph_full", [str(PRO / "graph_e1f22.py"), "--mode", "full"])
        rc |= run_step("dense_full", [str(PRO / "ervf_dense.py"), "--mode", "full"])
        graph_status = result_status("PRO_G0_E1F22_GRAPH_AB.json")
        if graph_status == "pass":
            rc |= run_step("epoch_full", [str(PRO / "epoch_graph.py"), "--mode", "full"])
        else:
            note = {
                "status": "dependency_skipped",
                "reason": f"PRO_G0 status is {graph_status!r}; run epoch manually only after reviewing graph correctness",
            }
            path = result_path("PRO_G2_DEPENDENCY_SKIP.json")
            path.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
            print(f"Skipping epoch graph: {note['reason']}")
    elif args.mode == "verify":
        return run_step("verify", [str(PRO / "verify_results.py")])
    elif args.mode == "report":
        return run_step("report", [str(PRO / "build_report.py")])

    # Always leave a machine verdict and a human report after experimental runs.
    rc |= run_step("verify", [str(PRO / "verify_results.py")])
    rc |= run_step("report", [str(PRO / "build_report.py")])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
