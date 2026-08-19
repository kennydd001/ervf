from __future__ import annotations

import json
from pathlib import Path
from common import REPO, write_json_atomic, utc_now

R = REPO / "pro_research" / "results" / "s100_phase14n2"
REF = R / "reference"

def load_evidence():
    out = {}
    if not REF.exists():
        return out
    for p in REF.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        name = str(p.relative_to(REF)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix.lower() == ".json":
                try:
                    out[name] = {"data": json.loads(text), "text": text}
                except Exception:
                    out[name] = {"data": None, "text": text}
            else:
                out[name] = {"data": None, "text": text}
        except Exception:
            pass
    return out

def strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, (str, bool, int, float)):
        yield str(obj)

def main():
    R.mkdir(parents=True, exist_ok=True)
    files = load_evidence()

    c0 = None
    c1 = []
    c2d = None
    for name, record in files.items():
        low = name.lower()
        data = record.get("data")
        text = record.get("text", "")
        if "c0b" in low and ("format" in low or "audit" in low):
            c0 = (name, data, text)
        if ("c1" in low or "repack" in low) and "verify" not in low:
            c1.append((name, data, text))
        if "c2d_m_scaling" in low and isinstance(data, dict):
            c2d = (name, data, text)

    format_green = None
    if c0 is not None:
        blob = (((" ".join(strings(c0[1]))) if c0[1] is not None else "")
                + " " + c0[2]).lower()
        format_green = (
            "exact" in blob
            and ("group16" in blob or "group-16" in blob or "packed" in blob)
        )

    repack_green = None
    repack_sources = []
    for name, data, text in c1:
        blob = (((" ".join(strings(data))) if data is not None else "")
                + " " + text).lower()
        if any(x in blob for x in ("lossless", "bitexact", "bit-exact", "exact")):
            if not any(x in blob for x in ("technical_failure", '"fail"', " failed ")):
                repack_green = True
                repack_sources.append(name)
    if c1 and repack_green is None:
        repack_green = False

    free_m_green = None
    c2d_detail = None
    if c2d is not None and c2d[1].get("status") == "measured":
        arms = c2d[1].get("arms", {})
        rows = []
        for name, rec in arms.items():
            rows.append({
                "shape": name,
                "rotation_over_l2": rec.get("working_set_over_l2"),
                "M4_over_M1": rec.get("M4_over_M1"),
                "M8_over_M1": rec.get("M8_over_M1"),
                "M4_speedup_per_token_vs_ervf": rec.get(
                    "M4_speedup_per_token_vs_ervf"
                ),
                "M8_speedup_per_token_vs_ervf": rec.get(
                    "M8_speedup_per_token_vs_ervf"
                ),
            })
        cold = [
            x for x in rows
            if x["rotation_over_l2"] is not None
            and x["rotation_over_l2"] >= 4.0
        ]
        m4 = [
            x for x in cold
            if x["M4_over_M1"] is not None
            and x["M4_over_M1"] <= 1.30
        ]
        m8 = [
            x for x in cold
            if x["M8_over_M1"] is not None
            and x["M8_over_M1"] <= 1.45
        ]
        free_m_green = len(cold) >= 4 and len(m4) >= 3 and len(m8) >= 3
        c2d_detail = {
            "source": c2d[0],
            "rows": rows,
            "cold_shapes": len(cold),
            "M4_green_shapes": len(m4),
            "M8_green_shapes": len(m8),
        }

    complete = all(x is not None for x in (
        format_green, repack_green, free_m_green
    ))
    open_c3 = (
        bool(format_green and repack_green and free_m_green)
        if complete else None
    )

    out = {
        "kind": "s100_phase14n2_adjudication",
        "created_utc": utc_now(),
        "reference_evidence_file_count": len(files),
        "FORMAT_GROUP16_PACKED_GREEN": format_green,
        "LOSSLESS_REPACK_EVIDENCE_GREEN": repack_green,
        "repack_sources": repack_sources,
        "NATIVE_FP4_FREE_M_GREEN": free_m_green,
        "c2d": c2d_detail,
        "instrumentation_complete": complete,
        "NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN": open_c3,
        "s100_single_achieved": False,
        "claim_boundary": (
            "authorizes real-weight C3 implementation only; "
            "N2 does not execute real repacked weights natively"
        ),
    }
    write_json_atomic(R / "S100_PHASE14N2_SUMMARY.json", out, archive=True)
    text = (
        "S100 PHASE 14N2 — NATIVE NVFP4 / SM120\n"
        f"FORMAT_GROUP16_PACKED_GREEN: {format_green}\n"
        f"LOSSLESS_REPACK_EVIDENCE_GREEN: {repack_green}\n"
        f"NATIVE_FP4_FREE_M_GREEN: {free_m_green}\n"
        f"NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN: {open_c3}\n"
        f"Instrumentation complete: {complete}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (R / "S100_PHASE14N2_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
