"""Independent CPU-only verifier for PRO V12C raw output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "pro_research" / "results" / "v12_async" / "PRO_V12C_EVENT_WAIT.json"
OUT = REPO / "pro_research" / "results" / "v12_async" / "PRO_V12C_VERIFY.json"


def divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y): return i
    return None if len(a) == len(b) else min(len(a), len(b))


def pct(xs: list[float], q: int) -> float | None:
    return None if not xs else float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def main() -> int:
    if not RESULT.exists():
        out = {"status": "missing", "result": str(RESULT)}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2)); return 2
    raw: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    mode = raw.get("mode")
    pp = raw.get("per_prompt", {})
    windows = [int(x) for x in raw.get("config", {}).get("windows", [])]
    sa = raw.get("sync", {}).get("A", {}).get("p50")
    sb = raw.get("sync", {}).get("B", {}).get("p50")
    drift = None if sa is None or sb is None else abs(float(sa) - float(sb))
    sync_parity = {name: divergence([int(x) for x in r.get("sync_a_ids", [])],
                                    [int(x) for x in r.get("sync_b_ids", [])])
                   for name, r in pp.items()}
    windows_out: dict[str, Any] = {}
    for w in windows:
        exact, safe, toks, ms = True, True, 0, 0.0
        divs: dict[str, int | None] = {}; p50s: dict[str, float | None] = {}
        for name, r in pp.items():
            ref = [int(x) for x in r.get("sync_a_ids", [])]
            c = r.get("event_wait", {}).get(str(w), {})
            d = divergence(ref, [int(x) for x in c.get("ids", [])])
            divs[name] = d; exact = exact and d is None
            mo = int(c.get("max_outstanding", 0))
            safe = safe and bool(c.get("safety_ok", False)) and mo <= w
            toks += int(c.get("decode_tokens", 0)); ms += float(c.get("total_ms", 0.0))
            gaps = [float(x) for x in c.get("raw_delivery_gap_ms", [])]
            drop = min(w, len(gaps)//4); p50s[name] = pct(gaps[drop:], 50)
        tps = 1000.0*toks/ms if ms > 0 else None
        finite = [float(x) for x in p50s.values() if x is not None]
        maxp = max(finite) if finite else None
        signal = bool(exact and safe and tps is not None and tps >= 50.0 and maxp is not None and maxp <= 20.0)
        verified = bool(mode == "full" and signal and drift is not None and drift <= 1.0 and toks >= 500)
        windows_out[str(w)] = {"exact": exact, "first_divergence": divs, "safety": safe,
                               "tokens": toks, "total_ms": ms, "tok_s_recomputed": tps,
                               "prompt_steady_p50_gap_ms_recomputed": p50s,
                               "max_prompt_steady_p50_gap_ms": maxp,
                               "E50_event_wait_signal_recomputed": signal,
                               "E50_event_wait_verified_recomputed": verified if mode == "full" else None}
    all_sync = all(v is None for v in sync_parity.values()) if sync_parity else False
    all_exact = all(v["exact"] for v in windows_out.values()) if windows_out else False
    all_safe = all(v["safety"] for v in windows_out.values()) if windows_out else False
    disagreements: list[str] = []
    runner = raw.get("event_wait_summary", {})
    for k, v in windows_out.items():
        r = runner.get(k, {})
        if r and bool(r.get("exact")) != bool(v["exact"]): disagreements.append(f"{k}:exact")
        if r and bool(r.get("safety_ok")) != bool(v["safety"]): disagreements.append(f"{k}:safety")
        if r and r.get("tok_s") is not None and v["tok_s_recomputed"] is not None and abs(float(r["tok_s"])-float(v["tok_s_recomputed"])) > 1e-9: disagreements.append(f"{k}:tok_s")
        if r and bool(r.get("E50_event_wait_signal")) != bool(v["E50_event_wait_signal_recomputed"]): disagreements.append(f"{k}:signal")
    out = {"kind": "pro_v12c_independent_verify",
           "status": "pass" if all_sync and all_exact and all_safe and not disagreements else "fail",
           "mode": mode, "sync_first_divergence": sync_parity, "p50_drift_ms_recomputed": drift,
           "baseline_drift_le_1ms": bool(drift is not None and drift <= 1.0),
           "windows": windows_out, "runner_verifier_disagreements": disagreements,
           "E50_event_wait_verified_any_recomputed": (any(bool(v.get("E50_event_wait_verified_recomputed")) for v in windows_out.values()) if mode == "full" else None)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8"); tmp.replace(OUT)
    print(json.dumps(out, indent=2)); return 0 if out["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
