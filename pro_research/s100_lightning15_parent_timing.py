from __future__ import annotations

import argparse
import json
import traceback

from common import REPO, percentiles, write_json_atomic, utc_now
from diag_component_marginals_graph import _run, _prefill, _reset_exact_state
from graph_e1f22 import _load_prompt_set
from s100_phase10a_runtime import build
from s100_lightning15_common import RESULTS, ensure_results, identity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("A", "B"), required=True)
    args = parser.parse_args()
    ensure_results()
    out = RESULTS / f"S100_LIGHTNING15_PARENT_TIMING_{args.role}.json"
    payload = {
        "kind": "s100_lightning15_parent_timing",
        "status": "started",
        "role": args.role,
        "started_utc": utc_now(),
    }
    try:
        bundle = build()
        rt = bundle.rt
        prompts, _e, length, _c = _load_prompt_set("full")
        length = max(256, int(length))
        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(96):
            rt.step_graph(None)
        rt._graph_stream.synchronize()

        raw = []
        ids = {}
        for prompt in prompts:
            output, timings = _run(
                rt, prompt["prompt_ids"], length
            )
            ids[prompt["prompt"]] = [int(x) for x in output]
            raw.extend(float(x) for x in timings)
        rt._graph_stream.synchronize()
        timing = percentiles(raw)
        payload.update({
            "status": "measured",
            "identity": identity(),
            "timing": timing,
            "tok_s": 1000.0 / float(timing["p50"]),
            "ids": ids,
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
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "role": args.role,
        "timing": payload.get("timing"),
        "tok_s": payload.get("tok_s"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
