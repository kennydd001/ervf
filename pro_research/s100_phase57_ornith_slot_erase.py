"""Phase57 test of llama-server slot erase as DFlash state mitigation."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase53_ornith_llama_dflash_e2e import (
    PROMPTS,
    _accepted_count,
    _drafted_count,
    _http_json,
    _start_server,
    _stop_server,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase57"
PREREG = REPO / "pro_research" / "S100_PHASE57_ORNITH_SLOT_ERASE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase57_ornith_slot_erase.py"
PHASE56 = RESULTS.parent / "s100_phase56" / "S100_PHASE56_ORNITH_UB256_CONFIRMATION.json"


def _request(port: int, prompt: str) -> tuple[dict[str, Any], float]:
    body = {
        "model": "ornith",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 5300,
        "max_tokens": 64,
        "cache_prompt": False,
        "id_slot": 0,
        "stream": False,
    }
    begin = time.perf_counter()
    response = _http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=1800
    )
    return response, time.perf_counter() - begin


def _record(name: str, response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    choice = response["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    reasoning = choice.get("message", {}).get("reasoning_content") or ""
    completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "wall_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "finish_reason": choice.get("finish_reason"),
        "text": reasoning + content,
        "response_timings": response.get("timings"),
    }


def _run_replicate(
    name: str,
    server: Path,
    target: Path,
    draft: Path,
    port: int,
    gpu_layers: int,
) -> dict[str, Any]:
    process = None
    thread = None
    logs: list[str] = []
    args: list[str] = []
    try:
        process, logs, thread, args = _start_server(
            server,
            target,
            draft,
            port,
            gpu_layers,
            startup_timeout=900,
            spec_k=8,
            extra_args=(
                "--no-cache-prompt", "--ubatch-size", "256", "--slots",
                "--slot-save-path", str(RESULTS / "slot_cache"),
            ),
        )
        coding_response, coding_elapsed = _request(port, PROMPTS[0]["text"])
        erase_response = _http_json(
            f"http://127.0.0.1:{port}/slots/0?action=erase", {}, timeout=60
        )
        arithmetic_response, arithmetic_elapsed = _request(port, PROMPTS[1]["text"])
        records = [
            _record("coding", coding_response, coding_elapsed),
            _record("arithmetic", arithmetic_response, arithmetic_elapsed),
        ]
        evidence = [
            line for line in logs
            if re.search(r"accept|draft|specu|timing|erase", line, flags=re.IGNORECASE)
        ]
        arm = {
            "status": "served",
            "name": name,
            "command": args,
            "erase_response": erase_response,
            "records": records,
            "evidence_log_lines": evidence[-300:],
            "log_tail": logs[-100:],
        }
        arm["accepted"] = _accepted_count(arm)
        arm["drafted"] = _drafted_count(arm)
        return arm
    finally:
        if process is not None and thread is not None:
            _stop_server(process, thread)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=10)
    parser.add_argument("--port-base", type=int, default=18130)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE57_ORNITH_SLOT_ERASE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase57_ornith_slot_erase",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        for path in (args.server, args.target, args.draft, PHASE56):
            if not path.is_file():
                raise FileNotFoundError(path)
        phase56 = json.loads(PHASE56.read_text(encoding="utf-8"))
        baseline = {
            row["name"]: row["text"]
            for row in phase56["arms"]["baseline_r1"]["records"]
        }
        version = subprocess.run(
            [str(args.server), "--version"],
            cwd=args.server.parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        arms = {
            f"slot_erase_r{replicate}": _run_replicate(
                f"slot_erase_r{replicate}",
                args.server.resolve(),
                args.target.resolve(),
                args.draft.resolve(),
                args.port_base + replicate - 1,
                args.gpu_layers,
            )
            for replicate in (1, 2)
        }
        exact = {
            f"{arm_name}:{row['name']}": row["text"] == baseline[row["name"]]
            for arm_name, arm in arms.items()
            for row in arm["records"]
        }
        all_serve = all(
            arm["status"] == "served"
            and all(row["completion_tokens"] > 0 and bool(row["text"]) for row in arm["records"])
            for arm in arms.values()
        )
        erase_ok = all(
            isinstance(arm.get("erase_response"), dict)
            and arm["erase_response"].get("id_slot") == 0
            and arm["erase_response"].get("n_erased") is not None
            for arm in arms.values()
        )
        positive_accept = all(
            arm.get("accepted") is not None and arm["accepted"] > 0
            for arm in arms.values()
        )
        gates = {
            "P57_G1_both_processes_serve_nonempty": all_serve,
            "P57_G2_both_coding_outputs_match_baseline": all(
                exact[f"slot_erase_r{replicate}:coding"] for replicate in (1, 2)
            ),
            "P57_G3_both_slot_erase_calls_succeed": erase_ok,
            "P57_G4_both_post_erase_arithmetic_outputs_match_baseline": all(
                exact[f"slot_erase_r{replicate}:arithmetic"] for replicate in (1, 2)
            ),
            "P57_G5_both_replicates_accept_positive": positive_accept,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "llama_version": version,
            "arms": arms,
            "exact_vs_phase56_baseline": exact,
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG))
    write_json_atomic(out, payload, archive=True)
    arms = payload.get("arms") or {}
    print(json.dumps({
        "status": payload.get("status"),
        "exact": payload.get("exact_vs_phase56_baseline"),
        "erase": {name: arm.get("erase_response") for name, arm in arms.items()},
        "acceptance": {
            name: {"accepted": arm.get("accepted"), "drafted": arm.get("drafted")}
            for name, arm in arms.items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
