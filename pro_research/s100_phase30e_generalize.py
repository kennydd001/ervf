from __future__ import annotations

import json

from common import utc_now, write_json_atomic
from s100_phase30e_common import RESULTS


def load(context: int, arm: str) -> dict:
    path = RESULTS / f"S100_PHASE30E_CTX{context}_{arm.upper()}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "measured" or not data.get("correctness_green"):
        raise RuntimeError(f"invalid context={context} arm={arm}")
    return data


def main() -> int:
    rows = []
    for context in (128, 4096):
        parent, candidate = load(context, "parent"), load(context, "candidate")
        p = float(parent["summary"]["median_ms"])
        c = float(candidate["summary"]["median_ms"])
        rows.append(
            {
                "context": context,
                "parent_ms_per_h4": p,
                "candidate_ms_per_h4": c,
                "gain_pct": (p - c) / p * 100.0,
                "parent_target_only_tok_s": 4000.0 / p,
                "candidate_target_only_tok_s": 4000.0 / c,
                "all_token_exact": True,
            }
        )
    primary = json.loads(
        (RESULTS / "S100_PHASE30E_ADJUDICATION.json").read_text(encoding="utf-8")
    )
    green = bool(
        primary.get("PHASE30E_ADOPTED")
        and all(row["all_token_exact"] and row["gain_pct"] > 0.0 for row in rows)
    )
    payload = {
        "kind": "s100_phase30e_generalization",
        "status": "measured",
        "created_utc": utc_now(),
        "contexts": rows,
        "primary_ctx1024_adopted": bool(primary.get("PHASE30E_ADOPTED")),
        "PHASE30E_GENERALIZATION_GREEN": green,
    }
    out = RESULTS / "S100_PHASE30E_GENERALIZATION.json"
    write_json_atomic(out, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if green else 2


if __name__ == "__main__":
    raise SystemExit(main())
