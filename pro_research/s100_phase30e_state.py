from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import expected_for_block, load_trace, prefill_to, release
from s100_phase24_common import make_synth
from s100_phase27_common import capture_arrays, compare_npz, phase27_gate
from s100_phase30e_common import RESULTS, make_candidate

CTX = 1024


def capture(mode: str) -> dict:
    json_path = RESULTS / f"S100_PHASE30E_STATE_{mode.upper()}.json"
    npz_path = RESULTS / f"S100_PHASE30E_STATE_{mode.upper()}.npz"
    payload = {
        "kind": "s100_phase30e_state_capture",
        "status": "started",
        "mode": mode,
        "context": CTX,
        "started_utc": utc_now(),
    }
    runtime = None
    try:
        cfg, _, _ = phase27_gate()
        tokens = load_trace()["tokens"]
        if mode == "parent":
            runtime, graph, keep = make_synth(CTX, cfg)
        else:
            runtime, graph, keep = make_candidate(CTX)
        capture_info = graph.setup_graph()

        def run_once():
            runtime.reset()
            prefill_to(runtime, tokens, CTX)
            graph.prepare_after_prefill()
            draft, expected = expected_for_block(tokens, CTX)
            ids = graph.launch(draft.tolist())
            if not np.array_equal(ids, expected):
                raise RuntimeError(
                    f"{mode} ids mismatch got={ids.tolist()} expected={expected.tolist()}"
                )
            return np.asarray(ids, np.int32), expected, graph.v.logits

        ids, expected, logits = run_once()
        repeat = None
        if mode == "candidate":
            repeat, expected2, _ = run_once()
            if not np.array_equal(repeat, expected2):
                raise RuntimeError("candidate deterministic replay mismatch")
            ids, expected, logits = run_once()

        arrays = capture_arrays(
            runtime, logits, CTX + 4, ids, ids_repeat=repeat
        )
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **arrays)
        payload.update(
            {
                "status": "measured",
                "capture_info": capture_info,
                "ids": ids.tolist(),
                "ids_repeat": None if repeat is None else repeat.tolist(),
                "expected": expected.tolist(),
                "array_count": len(arrays),
                "npz": str(npz_path),
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )
    finally:
        if runtime is not None:
            try:
                release(runtime)
            except Exception:
                pass
    write_json_atomic(json_path, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return payload


def compare() -> dict:
    parent = json.loads(
        (RESULTS / "S100_PHASE30E_STATE_PARENT.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (RESULTS / "S100_PHASE30E_STATE_CANDIDATE.json").read_text(encoding="utf-8")
    )
    if parent.get("status") != "measured" or candidate.get("status") != "measured":
        raise RuntimeError("Phase30E state captures are incomplete")
    state, gates = compare_npz(
        RESULTS / "S100_PHASE30E_STATE_PARENT.npz",
        RESULTS / "S100_PHASE30E_STATE_CANDIDATE.npz",
    )
    payload = {
        "kind": "s100_phase30e_state_check",
        "status": "measured",
        "created_utc": utc_now(),
        "state": state,
        "gates": gates,
        "parent_ids": parent.get("ids"),
        "candidate_ids": candidate.get("ids"),
        "candidate_repeat_ids": candidate.get("ids_repeat"),
        "PHASE30E_STATE_GREEN": bool(all(gates.values())),
    }
    out = RESULTS / "S100_PHASE30E_STATE_CHECK.json"
    write_json_atomic(out, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "candidate", "compare"), required=True)
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare()
        return 0 if result.get("PHASE30E_STATE_GREEN") else 2
    result = capture(args.mode)
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
