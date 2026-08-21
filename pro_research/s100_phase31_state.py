from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import expected_for_block, load_trace, prefill_to, release
from s100_phase27_common import capture_arrays, compare_npz
from s100_phase30e_common import make_candidate as make_parent
from s100_phase31_common import (
    RESULTS,
    make_attention_head_direct_candidate,
    make_attention_direct_candidate,
    make_candidate,
    make_dense_direct_candidate,
    make_group_down_candidate,
    make_staged_candidate,
)


CTX = 1024
CAPTURE_MODES = (
    "parent",
    "sink",
    "reduce_sink",
    "staged",
    "group_down",
    "attention_direct",
    "dense_direct",
    "attention_head_m4",
    "attention_head_m2",
)


def capture(mode: str) -> dict:
    json_path = RESULTS / f"S100_PHASE31_STATE_{mode.upper()}.json"
    npz_path = RESULTS / f"S100_PHASE31_STATE_{mode.upper()}.npz"
    payload = {
        "kind": "s100_phase31_state_capture",
        "status": "started",
        "mode": mode,
        "context": CTX,
        "started_utc": utc_now(),
    }
    runtime = None
    try:
        tokens = load_trace()["tokens"]
        if mode == "parent":
            runtime, graph, keep = make_parent(CTX)
        elif mode == "staged":
            runtime, graph, keep = make_staged_candidate(CTX)
        elif mode == "group_down":
            runtime, graph, keep = make_group_down_candidate(CTX)
        elif mode == "attention_direct":
            runtime, graph, keep = make_attention_direct_candidate(CTX)
        elif mode == "dense_direct":
            runtime, graph, keep = make_dense_direct_candidate(CTX)
        elif mode.startswith("attention_head_"):
            runtime, graph, keep = make_attention_head_direct_candidate(
                CTX, head_mode=mode.rsplit("_", 1)[-1]
            )
        else:
            runtime, graph, keep = make_candidate(CTX, mode=mode)
        capture_info = graph.setup_graph()

        def run_once():
            runtime.reset()
            prefill_to(runtime, tokens, CTX)
            graph.prepare_after_prefill()
            drafts, expected = expected_for_block(tokens, CTX)
            ids = graph.launch(drafts.tolist())
            if not np.array_equal(ids, expected):
                raise RuntimeError(
                    f"{mode} ids mismatch got={ids.tolist()} expected={expected.tolist()}"
                )
            return np.asarray(ids, np.int32), expected, graph.v.logits

        ids, expected, logits = run_once()
        repeat = None
        if mode != "parent":
            repeat, expected2, _ = run_once()
            if not np.array_equal(repeat, expected2):
                raise RuntimeError(f"{mode} deterministic replay mismatch")
            ids, expected, logits = run_once()

        arrays = capture_arrays(runtime, logits, CTX + 4, ids, ids_repeat=repeat)
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


def compare(modes=None) -> dict:
    if modes is None:
        modes = ("sink", "reduce_sink", "staged", "group_down", "attention_direct")
    parent_json = json.loads(
        (RESULTS / "S100_PHASE31_STATE_PARENT.json").read_text(encoding="utf-8")
    )
    arms = {}
    all_green = True
    for mode in modes:
        candidate_json = json.loads(
            (RESULTS / f"S100_PHASE31_STATE_{mode.upper()}.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            parent_json.get("status") != "measured"
            or candidate_json.get("status") != "measured"
        ):
            raise RuntimeError(f"Phase31 {mode} state captures are incomplete")
        state, gates = compare_npz(
            RESULTS / "S100_PHASE31_STATE_PARENT.npz",
            RESULTS / f"S100_PHASE31_STATE_{mode.upper()}.npz",
        )
        green = bool(all(gates.values()))
        all_green = all_green and green
        arms[mode] = {
            "state": state,
            "gates": gates,
            "ids": candidate_json.get("ids"),
            "repeat_ids": candidate_json.get("ids_repeat"),
            "green": green,
        }

    payload = {
        "kind": "s100_phase31_state_check",
        "status": "measured",
        "created_utc": utc_now(),
        "parent_ids": parent_json.get("ids"),
        "arms": arms,
        "PHASE31_STATE_GREEN": all_green,
    }
    write_json_atomic(RESULTS / "S100_PHASE31_STATE_CHECK.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            *CAPTURE_MODES,
            "compare",
            "compare_attention",
            "compare_head",
        ),
        required=True,
    )
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare()
        return 0 if result.get("PHASE31_STATE_GREEN") else 2
    if args.mode == "compare_attention":
        result = compare(("attention_direct",))
        return 0 if result.get("PHASE31_STATE_GREEN") else 2
    if args.mode == "compare_head":
        result = compare(("attention_head_m4",))
        return 0 if result.get("PHASE31_STATE_GREEN") else 2
    result = capture(args.mode)
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
