from __future__ import annotations

import argparse
import json
import traceback

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    assert_lightning,
    evaluate_runtime,
    normalize_eager_moe,
)
from s100_lightning16r_common import (
    candidate_signature,
    canonical_cases,
    ensure_results,
    quality_path,
)
from s100_lightning16r_native import (
    CUBLAS_WARMUP_DIAG,
    HANDOFFS,
    PointerDispatch,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--terms", type=int, choices=(1, 2), required=True
    )
    parser.add_argument("--cases-json", required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "validation", "heldout"),
        required=True,
    )
    parser.add_argument(
        "--handoff",
        choices=tuple(sorted(HANDOFFS)),
        required=True,
    )
    args = parser.parse_args()

    ensure_results()
    cases = canonical_cases(
        json.loads(args.cases_json)
    )
    output = quality_path(args.name, args.split)
    payload = {
        "kind": "s100_lightning16r_quality",
        "status": "started",
        "name": args.name,
        "terms": int(args.terms),
        "cases": cases,
        "handoff": args.handoff,
        "split": args.split,
        "candidate_signature": candidate_signature(
            terms=args.terms,
            cases=cases,
            handoff=args.handoff,
        ),
        "started_utc": utc_now(),
    }
    bundle = None
    try:
        identity = assert_lightning()
        bundle = build()
        runtime = bundle.rt
        runtime._graph = None
        runtime.graph_mode = False
        normalize_eager_moe(runtime)
        dispatch = PointerDispatch(
            runtime,
            terms=args.terms,
            handoff=args.handoff,
            enabled_cases=set(cases),
        ).install()
        result = evaluate_runtime(
            runtime,
            split=args.split,
            deterministic=args.split == "heldout",
        )
        payload.update({
            "status": "measured",
            "identity": identity,
            "dispatch": {
                "native_calls": dispatch.native_calls,
                "original_calls": dispatch.original_calls,
                "sync_calls": dispatch.sync_calls,
                "paired_sync_elisions": (
                    dispatch.paired_sync_elisions
                ),
                "torch_mm_style": (
                    dispatch.engine.mm.style
                ),
                "cublas_warmup": dict(CUBLAS_WARMUP_DIAG),
            },
            **result,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(output, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "name": payload["name"],
        "split": payload["split"],
        "summary": payload.get("summary"),
        "strict_pass": payload.get("strict_pass"),
        "official_pass": payload.get("official_pass"),
        "dispatch": payload.get("dispatch"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
