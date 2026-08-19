from __future__ import annotations

import argparse
import hashlib
import json
import traceback

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    RESULTS, assert_lightning, ensure_results,
    evaluate_runtime, normalize_eager_moe,
)
from s100_lightning16_native import CUBLAS_WARMUP_DIAG, PointerDispatch

def slug(name):
    return "".join(
        ch if ch.isalnum() else "_"
        for ch in name
    ).strip("_").upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--terms", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "validation", "heldout"),
        required=True,
    )
    parser.add_argument(
        "--handoff",
        choices=("legacy", "context_first", "sync_control"),
        default="context_first",
    )
    args = parser.parse_args()
    ensure_results()
    cases = json.loads(args.cases_json)
    if isinstance(cases, str):
        cases = [cases]
    if not isinstance(cases, list):
        raise TypeError("cases-json must decode to a list")
    signature = hashlib.sha256(
        json.dumps(
            {
                "terms": args.terms,
                "cases": sorted(cases),
                "handoff": args.handoff,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = RESULTS / (
        f"S100_LIGHTNING16_QUALITY_{slug(args.name)}_"
        f"{args.split.upper()}.json"
    )
    payload = {
        "kind": "s100_lightning16_quality",
        "status": "started",
        "name": args.name,
        "terms": args.terms,
        "cases": sorted(cases),
        "handoff": args.handoff,
        "split": args.split,
        "candidate_signature": signature,
        "started_utc": utc_now(),
    }
    try:
        ident = assert_lightning()
        bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)
        dispatch = PointerDispatch(
            rt,
            terms=args.terms,
            handoff=args.handoff,
            enabled_cases=set(cases),
        ).install()
        result = evaluate_runtime(
            rt,
            split=args.split,
            deterministic=args.split == "heldout",
        )
        payload.update({
            "status": "measured",
            "identity": ident,
            "dispatch": {
                "native_calls": dispatch.native_calls,
                "original_calls": dispatch.original_calls,
                "torch_mm_style": dispatch.engine.mm.style,
                "cublas_warmup": dict(CUBLAS_WARMUP_DIAG),
            },
            **result,
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
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

    write_json_atomic(OUT := output, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "name": args.name,
        "split": args.split,
        "summary": payload.get("summary"),
        "strict_pass": payload.get("strict_pass"),
        "official_pass": payload.get("official_pass"),
        "dispatch": payload.get("dispatch"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
