from __future__ import annotations
import json
from common import REPO, write_json_atomic, utc_now

R = REPO / "pro_research" / "results" / "s100_phase14d2"

def load(name):
    p = R / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    comp = load("S100_PHASE14D2_COMPONENT.json")
    val = load("S100_PHASE14D2_VALIDATION.json")
    hold = load("S100_PHASE14D2_HELDOUT.json")

    complete = bool(
        comp and comp.get("status") == "measured"
        and val and val.get("status") == "measured"
        and (
            not val.get("strict_pass")
            or (hold and hold.get("status") == "measured")
        )
    )

    if not comp or comp.get("status") != "measured":
        b1 = b4 = None
    elif not val or val.get("status") != "measured":
        b1 = b4 = None
    elif not val.get("strict_pass"):
        b1 = b4 = False
    elif not hold or hold.get("status") != "measured":
        b1 = b4 = None
    else:
        quality = bool(hold.get("official_pass"))
        b1 = bool(comp.get("B1_DIRECT_COMPONENT_PASS") and quality)
        b4 = bool(comp.get("B4_BLOCK_COMPONENT_PASS") and quality)

    out = {
        "kind": "s100_phase14d2_summary",
        "created_utc": utc_now(),
        "instrumentation_complete": complete,
        "NATIVE_BF16_B1_DIRECT_OPEN": b1,
        "NATIVE_BF16_BLOCK_BUILD_OPEN": b4,
        "component_B1": (
            (comp or {}).get("per_B", {}).get("1")
        ),
        "component_B4": (
            (comp or {}).get("per_B", {}).get("4")
        ),
        "validation": (val or {}).get("summary"),
        "heldout": (hold or {}).get("summary"),
        "s100_single_achieved": False,
        "claim_boundary": (
            "runtime-build authorization only; no production graph timing"
        ),
    }
    R.mkdir(parents=True, exist_ok=True)
    write_json_atomic(R / "S100_PHASE14D2_SUMMARY.json", out, archive=True)
    text = (
        "S100 PHASE 14D2 — NATIVE BF16\n"
        f"Instrumentation complete: {complete}\n"
        f"NATIVE_BF16_B1_DIRECT_OPEN: {b1}\n"
        f"NATIVE_BF16_BLOCK_BUILD_OPEN: {b4}\n"
        f"Validation strict: {(val or {}).get('strict_pass')}\n"
        f"Heldout official: {(hold or {}).get('official_pass')}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (R / "S100_PHASE14D2_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
