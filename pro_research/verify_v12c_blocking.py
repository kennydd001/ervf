"""Second independent V12C verifier: require every measured token arm to prove
that its delivery events were created with blocking-sync semantics."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "pro_research" / "results" / "v12_async"
RAW = DIR / "PRO_V12C_EVENT_WAIT.json"
BASE = DIR / "PRO_V12C_VERIFY.json"
OUT = DIR / "PRO_V12C_BLOCKING_VERIFY.json"


def main() -> int:
    if not RAW.exists() or not BASE.exists():
        rec = {"status": "missing", "raw_exists": RAW.exists(), "base_verify_exists": BASE.exists()}
    else:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
        base = json.loads(BASE.read_text(encoding="utf-8"))
        flags = {}
        all_blocking = True
        for prompt, p in raw.get("per_prompt", {}).items():
            flags[prompt] = {}
            for window, arm in p.get("event_wait", {}).items():
                value = arm.get("event_blocking_sync")
                flags[prompt][window] = value
                all_blocking = all_blocking and value is True
        rec = {"kind": "pro_v12c_blocking_semantics_verify",
               "status": "pass" if base.get("status") == "pass" and all_blocking else "fail",
               "base_verifier_status": base.get("status"),
               "all_event_arms_report_blocking_sync": all_blocking,
               "flags": flags}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0 if rec.get("status") == "pass" else 2

if __name__ == "__main__":
    raise SystemExit(main())
