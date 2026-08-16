"""Independent CPU verifier for PRO V12B.

This file does not import credit_stream_v12b.py or queue_stream_v12.py. It reads
only the raw JSON artifact and recomputes parity, timing summaries and E50 gates.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "pro_research" / "results" / "v12_async" / "PRO_V12B_CREDIT_STREAM.json"
OUT = REPO / "pro_research" / "results" / "v12_async" / "PRO_V12B_VERIFY.json"


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    if not RESULT.exists():
        payload = {"status": "missing", "result": str(RESULT)}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    raw: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    mode = raw.get("mode")
    per_prompt = raw.get("per_prompt", {})
    windows = [int(x) for x in raw.get("config", {}).get("windows", [])]

    sync_a_p50 = raw.get("sync", {}).get("A", {}).get("p50")
    sync_b_p50 = raw.get("sync", {}).get("B", {}).get("p50")
    if sync_a_p50 is None or sync_b_p50 is None:
        drift = None
        sync_mid = None
    else:
        drift = abs(float(sync_a_p50) - float(sync_b_p50))
        sync_mid = (float(sync_a_p50) + float(sync_b_p50)) / 2.0

    sync_parity: dict[str, Any] = {}
    for name, rec in per_prompt.items():
        a = [int(x) for x in rec.get("sync_a_ids", [])]
        b = [int(x) for x in rec.get("sync_b_ids", [])]
        div = first_divergence(a, b)
        sync_parity[name] = {"identical": div is None, "first_divergence": div}

    verified_windows: dict[str, Any] = {}
    for window in windows:
        total_tokens = 0
        total_ms = 0.0
        exact = True
        safety = True
        prompt_p50: dict[str, float | None] = {}
        prompt_p95: dict[str, float | None] = {}
        divs: dict[str, int | None] = {}
        max_outstanding = 0
        raw_gap_count = 0

        for name, rec in per_prompt.items():
            ref = [int(x) for x in rec.get("sync_a_ids", [])]
            crec = rec.get("credit", {}).get(str(window), {})
            ids = [int(x) for x in crec.get("ids", [])]
            div = first_divergence(ref, ids)
            divs[name] = div
            exact = exact and div is None
            safe_flag = bool(crec.get("safety_ok", False))
            outstanding = int(crec.get("max_outstanding", 0))
            safety = safety and safe_flag and outstanding <= window
            max_outstanding = max(max_outstanding, outstanding)
            total_tokens += int(crec.get("decode_tokens", 0))
            total_ms += float(crec.get("total_ms", 0.0))

            raw_gaps = [float(x) for x in crec.get("raw_delivery_gap_ms", [])]
            raw_gap_count += len(raw_gaps)
            drop = min(window, len(raw_gaps) // 4)
            steady = raw_gaps[drop:]
            prompt_p50[name] = pctl(steady, 50)
            prompt_p95[name] = pctl(steady, 95)

        tok_s = 1000.0 * total_tokens / total_ms if total_ms > 0 else None
        finite_p50 = [float(v) for v in prompt_p50.values() if v is not None]
        max_prompt_p50 = max(finite_p50) if finite_p50 else None
        signal = bool(
            exact and safety and tok_s is not None and tok_s >= 50.0
            and max_prompt_p50 is not None and max_prompt_p50 <= 20.0
        )
        enough = total_tokens >= 500
        full_verified = bool(
            mode == "full" and signal and drift is not None and drift <= 1.0 and enough
        )
        verified_windows[str(window)] = {
            "exact": exact,
            "first_divergence": divs,
            "safety": safety,
            "max_outstanding": max_outstanding,
            "tokens": total_tokens,
            "total_ms": total_ms,
            "tok_s_recomputed": tok_s,
            "raw_gap_count": raw_gap_count,
            "prompt_steady_p50_gap_ms_recomputed": prompt_p50,
            "prompt_steady_p95_gap_ms_recomputed": prompt_p95,
            "max_prompt_steady_p50_gap_ms": max_prompt_p50,
            "E50_streamed_signal_recomputed": signal,
            "full_tokens_ge_500": enough if mode == "full" else None,
            "E50_streamed_credit_verified_recomputed": full_verified if mode == "full" else None,
        }

    all_sync = all(v["identical"] for v in sync_parity.values()) if sync_parity else False
    all_credit = all(v["exact"] for v in verified_windows.values()) if verified_windows else False
    all_safe = all(v["safety"] for v in verified_windows.values()) if verified_windows else False
    any_signal = any(v["E50_streamed_signal_recomputed"] for v in verified_windows.values())
    any_full = any(bool(v.get("E50_streamed_credit_verified_recomputed")) for v in verified_windows.values())

    runner_summary = raw.get("credit_summary", {})
    disagreements: list[str] = []
    for k, v in verified_windows.items():
        rv = runner_summary.get(k, {})
        if rv:
            if bool(rv.get("exact")) != bool(v["exact"]):
                disagreements.append(f"window {k}: exact")
            if bool(rv.get("safety_ok")) != bool(v["safety"]):
                disagreements.append(f"window {k}: safety")
            rtps = rv.get("tok_s")
            vtps = v.get("tok_s_recomputed")
            if rtps is not None and vtps is not None and abs(float(rtps) - float(vtps)) > 1e-9:
                disagreements.append(f"window {k}: tok_s")
            if bool(rv.get("E50_streamed_signal")) != bool(v["E50_streamed_signal_recomputed"]):
                disagreements.append(f"window {k}: E50 signal")

    payload = {
        "kind": "pro_v12b_independent_verify",
        "status": "pass" if all_sync and all_credit and all_safe and not disagreements else "fail",
        "source_result_status": raw.get("status"),
        "mode": mode,
        "sync": {
            "parity": sync_parity,
            "p50_drift_ms_recomputed": drift,
            "p50_midpoint_ms_recomputed": sync_mid,
            "baseline_drift_le_1ms": bool(drift is not None and drift <= 1.0),
        },
        "windows": verified_windows,
        "all_credit_exact": all_credit,
        "all_safety": all_safe,
        "E50_streamed_signal_any_recomputed": any_signal,
        "E50_streamed_credit_verified_any_recomputed": any_full if mode == "full" else None,
        "runner_verifier_disagreements": disagreements,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
