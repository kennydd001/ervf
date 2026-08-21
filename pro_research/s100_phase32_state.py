from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, prefill_to, release
from s100_phase25_common import expected_for_h8
from s100_phase27_common import capture_arrays, compare_npz
from s100_phase32_common import ARMS, RESULTS, make_candidate, make_parent


CONTEXT = 1024


def capture(mode: str) -> dict:
    import cupy as cp

    label = mode.upper()
    json_path = RESULTS / f"S100_PHASE32_STATE_{label}.json"
    npz_path = RESULTS / f"S100_PHASE32_STATE_{label}.npz"
    payload = {
        "kind": "s100_phase32_state_capture",
        "status": "started",
        "mode": mode,
        "context": CONTEXT,
        "started_utc": utc_now(),
    }
    runtime = None
    try:
        tokens = load_trace()["tokens"]
        drafts, expected = expected_for_h8(tokens, CONTEXT)
        if mode == "parent":
            runtime, graph, keep = make_parent(CONTEXT)
        else:
            runtime, graph, keep = make_candidate(CONTEXT, mode)
        capture_info = graph.setup_graph()

        def run_once():
            runtime.reset()
            prefill_to(runtime, tokens, CONTEXT)
            graph.prepare_after_prefill()
            if mode == "parent":
                first = graph.launch(drafts[:4].tolist())
                logits_first = graph.v.logits.copy()
                second = graph.launch(drafts[4:].tolist())
                ids = np.concatenate([first, second])
                logits = cp.concatenate([logits_first, graph.v.logits], axis=0)
            else:
                ids = graph.launch(drafts.tolist())
                logits = graph.v.logits.copy()
            if not np.array_equal(ids, expected):
                raise RuntimeError(
                    f"{mode} ids mismatch got={ids.tolist()} expected={expected.tolist()}"
                )
            return np.asarray(ids, np.int32), logits

        ids, logits = run_once()
        repeat = ids.copy()
        if mode != "parent":
            repeat, _ = run_once()
            if not np.array_equal(repeat, expected):
                raise RuntimeError(f"{mode} deterministic replay mismatch")
            ids, logits = run_once()

        arrays = capture_arrays(
            runtime, logits, CONTEXT + 8, ids, ids_repeat=repeat,
        )
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **arrays)
        payload.update(
            {
                "status": "measured",
                "capture_info": capture_info,
                "ids": ids.tolist(),
                "ids_repeat": np.asarray(repeat).tolist(),
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


def compare(arm: str) -> dict:
    parent_meta = json.loads(
        (RESULTS / "S100_PHASE32_STATE_PARENT.json").read_text(encoding="utf-8")
    )
    candidate_meta = json.loads(
        (RESULTS / f"S100_PHASE32_STATE_{arm.upper()}.json").read_text(
            encoding="utf-8"
        )
    )
    if parent_meta.get("status") != "measured" or candidate_meta.get("status") != "measured":
        raise RuntimeError("Phase32 state captures are incomplete")
    state, gates = compare_npz(
        RESULTS / "S100_PHASE32_STATE_PARENT.npz",
        RESULTS / f"S100_PHASE32_STATE_{arm.upper()}.npz",
    )
    payload = {
        "kind": "s100_phase32_state_check",
        "status": "measured",
        "arm": arm,
        "created_utc": utc_now(),
        "state": state,
        "gates": gates,
        "parent_ids": parent_meta.get("ids"),
        "candidate_ids": candidate_meta.get("ids"),
        "PHASE32_STATE_GREEN": bool(all(gates.values())),
    }
    write_json_atomic(RESULTS / "S100_PHASE32_STATE_CHECK.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", *ARMS, "compare"), required=True)
    parser.add_argument("--arm", choices=ARMS, default="dense_m8")
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare(args.arm)
        return 0 if result.get("PHASE32_STATE_GREEN") else 2
    result = capture(args.mode)
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
