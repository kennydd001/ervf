"""Phase53 independent llama.cpp baseline versus DFlash end-to-end smoke."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase53"
PREREG = REPO / "pro_research" / "S100_PHASE53_ORNITH_LLAMA_DFLASH_E2E_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase53_ornith_llama_dflash_e2e.py"
PROMPTS = (
    {
        "name": "coding",
        "text": (
            "Write a compact Python function named merge_intervals that accepts a list "
            "of [start, end] integer pairs, merges overlaps, and returns sorted pairs. "
            "Explain the complexity briefly."
        ),
    },
    {
        "name": "arithmetic",
        "text": (
            "Solve step by step: a store discounts a 240 euro item by 15 percent, then "
            "adds 21 percent VAT to the discounted price. What is the final price?"
        ),
    },
)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _read_text(url: str, timeout: float = 5.0) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _start_server(
    server: Path,
    target: Path,
    draft: Path | None,
    port: int,
    gpu_layers: int,
    startup_timeout: float,
    spec_k: int = 8,
    extra_args: tuple[str, ...] = (),
) -> tuple[subprocess.Popen[str], list[str], threading.Thread, list[str]]:
    args = [
        str(server),
        "-m", str(target),
        "-c", "4096",
        "-ngl", str(gpu_layers),
        "-fit", "off",
        "-fa", "on",
        "-np", "1",
        "--metrics",
        "--jinja",
        "--alias", "ornith",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    args.extend(extra_args)
    if draft is not None:
        args.extend([
            "-md", str(draft),
            "-ngld", "all",
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", str(spec_k),
        ])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        args,
        cwd=server.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    logs: list[str] = []

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            logs.append(line.rstrip())

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited rc={process.returncode}: " + " | ".join(logs[-30:])
            )
        try:
            if _read_text(f"http://127.0.0.1:{port}/health", 2.0):
                return process, logs, thread, args
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"llama-server startup exceeded {startup_timeout}s: " + " | ".join(logs[-30:]))


def _stop_server(process: subprocess.Popen[str], thread: threading.Thread) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    thread.join(timeout=5)


def _request(port: int, text: str, max_tokens: int) -> tuple[dict[str, Any], float]:
    body = {
        "model": "ornith",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 5300,
        "max_tokens": max_tokens,
        "stream": False,
    }
    begin = time.perf_counter()
    response = _http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=1800
    )
    return response, time.perf_counter() - begin


def _run_arm(
    name: str,
    server: Path,
    target: Path,
    draft: Path | None,
    port: int,
    gpu_layers: int,
) -> dict[str, Any]:
    process = None
    thread = None
    logs: list[str] = []
    args: list[str] = []
    started = time.perf_counter()
    try:
        process, logs, thread, args = _start_server(
            server, target, draft, port, gpu_layers, startup_timeout=900
        )
        load_seconds = time.perf_counter() - started
        _request(port, "Reply with only the word READY.", 8)
        records = []
        for prompt in PROMPTS:
            response, elapsed = _request(port, prompt["text"], 64)
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            reasoning = choice.get("message", {}).get("reasoning_content") or ""
            rendered = reasoning + content
            completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
            records.append({
                "name": prompt["name"],
                "elapsed_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "wall_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
                "finish_reason": choice.get("finish_reason"),
                "text": rendered,
                "response_timings": response.get("timings"),
            })
        metrics = _read_text(f"http://127.0.0.1:{port}/metrics", 30.0)
        evidence_lines = [
            line for line in logs
            if re.search(r"accept|draft|specu|timing", line, flags=re.IGNORECASE)
        ]
        metric_lines = [
            line for line in metrics.splitlines()
            if re.search(r"accept|draft|specu|decode", line, flags=re.IGNORECASE)
        ]
        return {
            "status": "served",
            "name": name,
            "command": args,
            "load_seconds": load_seconds,
            "records": records,
            "geomean_wall_tok_s": _geomean([row["wall_tok_s"] for row in records]),
            "evidence_log_lines": evidence_lines[-300:],
            "evidence_metric_lines": metric_lines[-300:],
            "log_tail": logs[-100:],
        }
    finally:
        if process is not None and thread is not None:
            _stop_server(process, thread)


def _accepted_count(arm: dict[str, Any]) -> int | None:
    timing_values = []
    for row in arm.get("records", []):
        timings = row.get("response_timings") or {}
        for key in ("draft_n_accepted", "n_accept", "tokens_drafted_accepted"):
            value = timings.get(key)
            if value is not None:
                timing_values.append(int(value))
                break
    if timing_values:
        return sum(timing_values)
    text = "\n".join(
        arm.get("evidence_log_lines", []) + arm.get("evidence_metric_lines", [])
    )
    patterns = (
        r"(?:accepted|n_accept|tokens_accepted)[^0-9]{0,20}(\d+)",
        r"(\d+)[^\n]{0,30}(?:accepted tokens|tokens accepted)",
    )
    values = []
    for pattern in patterns:
        values.extend(int(value) for value in re.findall(pattern, text, flags=re.IGNORECASE))
    return max(values) if values else None


def _drafted_count(arm: dict[str, Any]) -> int | None:
    timing_values = []
    for row in arm.get("records", []):
        timings = row.get("response_timings") or {}
        for key in ("draft_n", "n_drafted", "tokens_drafted"):
            value = timings.get(key)
            if value is not None:
                timing_values.append(int(value))
                break
    if timing_values:
        return sum(timing_values)
    text = "\n".join(
        arm.get("evidence_log_lines", []) + arm.get("evidence_metric_lines", [])
    )
    values = [
        int(value)
        for value in re.findall(
            r"(?:generated|drafted|n_drafted|tokens_drafted)[^0-9]{0,20}(\d+)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=10)
    parser.add_argument("--port-base", type=int, default=18087)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE53_ORNITH_LLAMA_DFLASH_E2E.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase53_ornith_llama_dflash_e2e",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "server": str(args.server.resolve()),
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "gpu_layers": int(args.gpu_layers),
    }
    try:
        for path in (args.server, args.target, args.draft):
            if not path.is_file():
                raise FileNotFoundError(path)
        version = subprocess.run(
            [str(args.server), "--version"],
            cwd=args.server.parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        baseline = _run_arm(
            "baseline", args.server.resolve(), args.target.resolve(), None,
            args.port_base, args.gpu_layers,
        )
        dflash = _run_arm(
            "dflash_k8", args.server.resolve(), args.target.resolve(), args.draft.resolve(),
            args.port_base + 1, args.gpu_layers,
        )
        baseline_by_name = {row["name"]: row for row in baseline["records"]}
        dflash_by_name = {row["name"]: row for row in dflash["records"]}
        exact = {
            name: baseline_by_name[name]["text"].encode("utf-8")
            == dflash_by_name[name]["text"].encode("utf-8")
            for name in baseline_by_name
        }
        accepted = _accepted_count(dflash)
        drafted = _drafted_count(dflash)
        evidence_present = bool(
            dflash.get("evidence_log_lines") or dflash.get("evidence_metric_lines")
        )
        served = baseline.get("status") == "served" and dflash.get("status") == "served"
        nonempty = all(
            row["completion_tokens"] > 0 and bool(row["text"])
            for arm in (baseline, dflash)
            for row in arm["records"]
        )
        gates = {
            "P53_G1_both_arms_serve": served,
            "P53_G2_all_completions_nonempty": nonempty,
            "P53_G3_greedy_outputs_byte_exact": all(exact.values()),
            "P53_G4_acceptance_evidence_and_positive_accept": evidence_present and accepted is not None and accepted > 0,
            "P53_G5_dflash_geomean_tok_s_gt_baseline": (
                dflash["geomean_wall_tok_s"] > baseline["geomean_wall_tok_s"]
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "llama_version": version,
            "artifact_bytes": {
                "target": args.target.stat().st_size,
                "draft": args.draft.stat().st_size,
            },
            "baseline": baseline,
            "dflash": dflash,
            "greedy_output_exact_by_prompt": exact,
            "parsed_accepted_count": accepted,
            "parsed_drafted_count": drafted,
            "parsed_acceptance_rate": (
                accepted / drafted if accepted is not None and drafted else None
            ),
            "speedup_geomean": dflash["geomean_wall_tok_s"] / baseline["geomean_wall_tok_s"],
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
    print(json.dumps({
        "status": payload.get("status"),
        "baseline_tok_s": (payload.get("baseline") or {}).get("geomean_wall_tok_s"),
        "dflash_tok_s": (payload.get("dflash") or {}).get("geomean_wall_tok_s"),
        "speedup": payload.get("speedup_geomean"),
        "accepted": payload.get("parsed_accepted_count"),
        "drafted": payload.get("parsed_drafted_count"),
        "acceptance_rate": payload.get("parsed_acceptance_rate"),
        "exact": payload.get("greedy_output_exact_by_prompt"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
