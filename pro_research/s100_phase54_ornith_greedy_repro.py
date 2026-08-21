"""Phase54 fresh-process greedy reproducibility matrix for Ornith DFlash."""
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
    _accepted_count,
    _drafted_count,
    _http_json,
    _start_server,
    _stop_server,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase54"
PREREG = REPO / "pro_research" / "S100_PHASE54_ORNITH_GREEDY_REPRO_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase54_ornith_greedy_repro.py"
PROMPT = (
    "Solve step by step: a store discounts a 240 euro item by 15 percent, then "
    "adds 21 percent VAT to the discounted price. What is the final price?"
)


def _request(port: int) -> tuple[dict[str, Any], float]:
    body = {
        "model": "ornith",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 5300,
        "max_tokens": 64,
        "cache_prompt": False,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
    }
    begin = time.perf_counter()
    response = _http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=1800
    )
    return response, time.perf_counter() - begin


def _run_once(
    name: str,
    server: Path,
    target: Path,
    draft: Path | None,
    spec_k: int,
    port: int,
    gpu_layers: int,
    server_extra_args: tuple[str, ...] = (),
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
            spec_k=spec_k,
            extra_args=("--no-cache-prompt",) + server_extra_args,
        )
        load_seconds = time.perf_counter() - started
        response, elapsed = _request(port)
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        reasoning = choice.get("message", {}).get("reasoning_content") or ""
        text = reasoning + content
        completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
        record = {
            "name": name,
            "elapsed_seconds": elapsed,
            "completion_tokens": completion_tokens,
            "wall_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
            "finish_reason": choice.get("finish_reason"),
            "text": text,
            "response_timings": response.get("timings"),
            "logprobs": choice.get("logprobs"),
        }
        evidence = [
            line for line in logs
            if re.search(r"accept|draft|specu|timing", line, flags=re.IGNORECASE)
        ]
        arm = {
            "status": "served",
            "name": name,
            "command": args,
            "load_seconds": load_seconds,
            "records": [record],
            "evidence_log_lines": evidence[-300:],
            "log_tail": logs[-100:],
        }
        arm["accepted"] = _accepted_count(arm)
        arm["drafted"] = _drafted_count(arm)
        return arm
    finally:
        if process is not None and thread is not None:
            _stop_server(process, thread)


def _first_text_difference(left: str, right: str) -> dict[str, Any] | None:
    if left == right:
        return None
    limit = min(len(left), len(right))
    index = next((i for i in range(limit) if left[i] != right[i]), limit)
    lo = max(0, index - 40)
    hi = index + 80
    return {
        "character_index": index,
        "baseline_context": left[lo:hi],
        "candidate_context": right[lo:hi],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=10)
    parser.add_argument("--port-base", type=int, default=18090)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE54_ORNITH_GREEDY_REPRO.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase54_ornith_greedy_repro",
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
        specifications = (
            ("baseline_r1", None, 0),
            ("baseline_r2", None, 0),
            ("dflash_k1_r1", args.draft.resolve(), 1),
            ("dflash_k1_r2", args.draft.resolve(), 1),
            ("dflash_k8_r1", args.draft.resolve(), 8),
            ("dflash_k8_r2", args.draft.resolve(), 8),
        )
        arms: dict[str, dict[str, Any]] = {}
        for offset, (name, draft, spec_k) in enumerate(specifications):
            arms[name] = _run_once(
                name,
                args.server.resolve(),
                args.target.resolve(),
                draft,
                spec_k,
                args.port_base + offset,
                args.gpu_layers,
            )
        texts = {name: arm["records"][0]["text"] for name, arm in arms.items()}
        baseline = texts["baseline_r1"]
        baseline_repeat = texts["baseline_r2"] == baseline
        k1_repeat = texts["dflash_k1_r1"] == texts["dflash_k1_r2"]
        k8_repeat = texts["dflash_k8_r1"] == texts["dflash_k8_r2"]
        k1_exact = all(texts[f"dflash_k1_r{i}"] == baseline for i in (1, 2))
        k8_exact = all(texts[f"dflash_k8_r{i}"] == baseline for i in (1, 2))
        served_nonempty = all(
            arm["status"] == "served"
            and arm["records"][0]["completion_tokens"] > 0
            and bool(arm["records"][0]["text"])
            for arm in arms.values()
        )
        positive_accept = all(
            arms[f"dflash_k{k}_r{i}"]["accepted"] is not None
            and arms[f"dflash_k{k}_r{i}"]["accepted"] > 0
            for k in (1, 8)
            for i in (1, 2)
        )
        gates = {
            "P54_G1_all_cells_serve_nonempty": served_nonempty,
            "P54_G2_baseline_repeat_byte_exact": baseline_repeat,
            "P54_G3_dflash_k1_repeat_byte_exact": k1_repeat,
            "P54_G4_dflash_k8_repeat_byte_exact": k8_repeat,
            "P54_G5_dflash_k1_matches_baseline": k1_exact,
            "P54_G6_dflash_k8_matches_baseline": k8_exact,
            "P54_G7_all_dflash_cells_accept_positive": positive_accept,
        }
        repeatable = baseline_repeat and k1_repeat and k8_repeat
        if repeatable and k1_exact and k8_exact:
            adjudication = "lossless_reproduced"
        elif repeatable and k1_exact and not k8_exact:
            adjudication = "multi_token_verification_divergence"
        elif repeatable and not k1_exact:
            adjudication = "draft_model_path_divergence_at_k1"
        else:
            adjudication = "nondeterministic_or_inconclusive"
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "llama_version": version,
            "arms": arms,
            "exactness": {
                "baseline_repeat": baseline_repeat,
                "dflash_k1_repeat": k1_repeat,
                "dflash_k8_repeat": k8_repeat,
                "dflash_k1_vs_baseline": k1_exact,
                "dflash_k8_vs_baseline": k8_exact,
            },
            "first_differences": {
                name: _first_text_difference(baseline, text)
                for name, text in texts.items()
                if name != "baseline_r1"
            },
            "adjudication": adjudication,
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
        "adjudication": payload.get("adjudication"),
        "exactness": payload.get("exactness"),
        "dflash_evidence": {
            name: {"accepted": arm.get("accepted"), "drafted": arm.get("drafted")}
            for name, arm in arms.items()
            if name.startswith("dflash")
        },
        "tok_s": {
            name: arm["records"][0]["wall_tok_s"] for name, arm in arms.items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
