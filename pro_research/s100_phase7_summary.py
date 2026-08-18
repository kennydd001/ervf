
from __future__ import annotations
import json

from common import REPO, utc_now, write_json_atomic
from s100_phase7_common import EXPECTED

OUT = (
    REPO / "pro_research" / "results" / "S100_PHASE7_SUMMARY.json"
)
TXT = OUT.with_suffix(".txt")


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    heldout = load(
        REPO / "pro_research" / "results" / "S100_PHASE7_HELDOUT.json"
    ) or {}
    backend = load(
        REPO / "pro_research" / "results"
        / "S100_PHASE7_BACKEND_SELECT.json"
    ) or {"selected_backend": "legacy"}

    rows = []
    for name in EXPECTED:
        h = (heldout.get("results") or {}).get(name) or {}
        timing = load(
            REPO / "pro_research" / "results"
            / f"S100_PHASE7_TIMING_COMPARE_{name.upper()}.json"
        )
        rows.append(
            {
                "candidate": name,
                "fidelity": h.get("status"),
                "quality": h.get("summary"),
                "timing_status": timing.get("status") if timing else None,
                "timing": timing.get("summary") if timing else None,
            }
        )

    good = [
        r for r in rows
        if r["fidelity"] == "v18_fidelity_candidate"
        and r["timing_status"] == "fresh_timing_candidate"
    ]
    fastest = (
        min(
            good,
            key=lambda r: float(
                r["timing"]["candidate_midpoint_ms"]
            ),
        )
        if good
        else None
    )
    s100 = bool(
        fastest
        and float(fastest["timing"]["candidate_midpoint_ms"]) <= 10.0
    )
    payload = {
        "kind": "s100_phase7_summary",
        "created_utc": utc_now(),
        "selected_exact_backend": backend.get(
            "selected_backend", "legacy"
        ),
        "candidates": rows,
        "fastest_fidelity_green": fastest,
        "s100_single_achieved": s100,
    }
    write_json_atomic(OUT, payload, archive=True)

    lines = [
        "S100 PHASE 7 SUMMARY",
        f"Exact backend: {payload['selected_exact_backend']}",
        "",
        "candidate | fidelity | ms | tok/s | top1 | dCE | KL",
    ]
    for row in rows:
        q = row.get("quality") or {}
        t = row.get("timing") or {}
        lines.append(
            f"{row['candidate']} | {row['fidelity']} | "
            f"{t.get('candidate_midpoint_ms')} | "
            f"{t.get('candidate_tok_s')} | "
            f"{q.get('top1_agreement')} | "
            f"{q.get('mean_ce_delta')} | "
            f"{q.get('mean_coarse_kl')}"
        )
    lines.extend(
        [
            "",
            "FASTEST FIDELITY-GREEN: "
            + (fastest["candidate"] if fastest else "None"),
            f"S100 SINGLE ACHIEVED: {s100}",
        ]
    )
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
