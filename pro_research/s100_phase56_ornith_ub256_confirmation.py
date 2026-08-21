"""Phase56 repeated two-prompt confirmation of the Ornith ubatch-256 candidate."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
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


RESULTS = REPO / "pro_research" / "results" / "s100_phase56"
PREREG = REPO / "pro_research" / "S100_PHASE56_ORNITH_UB256_CONFIRMATION_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase56_ornith_ub256_confirmation.py"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _request(port: int, prompt: str, max_tokens: int) -> tuple[dict[str, Any], float]:
    body = {
        "model": "ornith",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 5300,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "stream": False,
    }
    begin = time.perf_counter()
    response = _http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=1800
    )
    return response, time.perf_counter() - begin


def _run_replicate(
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
            server,
            target,
            draft,
            port,
            gpu_layers,
            startup_timeout=900,
            spec_k=8,
            extra_args=("--no-cache-prompt", "--ubatch-size", "256"),
        )
        load_seconds = time.perf_counter() - started
        records = []
        for prompt in PROMPTS:
            response, elapsed = _request(port, prompt["text"], 64)
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            reasoning = choice.get("message", {}).get("reasoning_content") or ""
            completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
            records.append({
                "name": prompt["name"],
                "elapsed_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "wall_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
                "finish_reason": choice.get("finish_reason"),
                "text": reasoning + content,
                "response_timings": response.get("timings"),
            })
        evidence = [
            line for line in logs
            if re.search(r"accept|draft|specu|timing", line, flags=re.IGNORECASE)
        ]
        arm = {
            "status": "served",
            "name": name,
            "command": args,
            "load_seconds": load_seconds,
            "records": records,
            "geomean_wall_tok_s": _geomean([row["wall_tok_s"] for row in records]),
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
    parser.add_argument("--port-base", type=int, default=18120)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE56_ORNITH_UB256_CONFIRMATION.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase56_ornith_ub256_confirmation",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "gpu_layers": int(args.gpu_layers),
        "target_ubatch": 256,
        "dflash_k": 8,
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
        specifications = (
            ("baseline_r1", None),
            ("baseline_r2", None),
            ("dflash_r1", args.draft.resolve()),
            ("dflash_r2", args.draft.resolve()),
        )
        arms = {
            name: _run_replicate(
                name,
                args.server.resolve(),
                args.target.resolve(),
                draft,
                args.port_base + offset,
                args.gpu_layers,
            )
            for offset, (name, draft) in enumerate(specifications)
        }
        by_prompt = {
            arm_name: {row["name"]: row for row in arm["records"]}
            for arm_name, arm in arms.items()
        }
        baseline_repeat = {
            prompt["name"]: (
                by_prompt["baseline_r1"][prompt["name"]]["text"]
                == by_prompt["baseline_r2"][prompt["name"]]["text"]
            )
            for prompt in PROMPTS
        }
        dflash_repeat = {
            prompt["name"]: (
                by_prompt["dflash_r1"][prompt["name"]]["text"]
                == by_prompt["dflash_r2"][prompt["name"]]["text"]
            )
            for prompt in PROMPTS
        }
        dflash_exact = {
            f"{arm_name}:{prompt['name']}": (
                by_prompt[arm_name][prompt["name"]]["text"]
                == by_prompt["baseline_r1"][prompt["name"]]["text"]
            )
            for arm_name in ("dflash_r1", "dflash_r2")
            for prompt in PROMPTS
        }
        all_serve = all(
            arm["status"] == "served"
            and all(row["completion_tokens"] > 0 and bool(row["text"]) for row in arm["records"])
            for arm in arms.values()
        )
        positive_accept = all(
            arms[name]["accepted"] is not None and arms[name]["accepted"] > 0
            for name in ("dflash_r1", "dflash_r2")
        )
        baseline_speed = statistics.median(
            arms[name]["geomean_wall_tok_s"] for name in ("baseline_r1", "baseline_r2")
        )
        dflash_speed = statistics.median(
            arms[name]["geomean_wall_tok_s"] for name in ("dflash_r1", "dflash_r2")
        )
        gates = {
            "P56_G1_all_processes_serve_nonempty": all_serve,
            "P56_G2_baseline_repeats_exact": all(baseline_repeat.values()),
            "P56_G3_dflash_repeats_exact": all(dflash_repeat.values()),
            "P56_G4_all_dflash_outputs_match_baseline": all(dflash_exact.values()),
            "P56_G5_both_dflash_replicates_accept_positive": positive_accept,
            "P56_G6_dflash_median_geomean_tok_s_gt_baseline": dflash_speed > baseline_speed,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "llama_version": version,
            "arms": arms,
            "baseline_repeat_exact": baseline_repeat,
            "dflash_repeat_exact": dflash_repeat,
            "dflash_exact_vs_baseline": dflash_exact,
            "median_geomean_wall_tok_s": {
                "baseline": baseline_speed,
                "dflash": dflash_speed,
            },
            "speedup_median_geomean": dflash_speed / baseline_speed,
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
        "baseline_repeat": payload.get("baseline_repeat_exact"),
        "dflash_repeat": payload.get("dflash_repeat_exact"),
        "dflash_exact": payload.get("dflash_exact_vs_baseline"),
        "median_tok_s": payload.get("median_geomean_wall_tok_s"),
        "speedup": payload.get("speedup_median_geomean"),
        "acceptance": {
            name: {"accepted": arm.get("accepted"), "drafted": arm.get("drafted")}
            for name, arm in arms.items() if name.startswith("dflash")
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
