from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import (
    load_trace,
    prefill_to,
    expected_for_block,
    release,
)
from s100_phase24_common import (
    make_synth,
    selected_config,
)
from s100_phase28_common import (
    RESULTS,
    Arm,
    capture_arrays,
    compare_npz,
    make_arm,
    phase28_gate,
)

CONTEXT = 1024


def selected_arm() -> Arm:
    selection = json.loads(
        (
            RESULTS / "S100_PHASE28_SELECTION.json"
        ).read_text(encoding="utf-8")
    )
    name = selection.get("selected_arm")
    if not name:
        raise RuntimeError("Phase28 selected arm missing")
    if not selection.get("RUN_STATE_GATE"):
        raise RuntimeError("Phase28 state gate was not opened")
    return Arm(str(name))


def capture(mode: str) -> dict:
    out_json = (
        RESULTS
        / f"S100_PHASE28_STATE_{mode.upper()}.json"
    )
    out_npz = (
        RESULTS
        / f"S100_PHASE28_STATE_{mode.upper()}.npz"
    )

    payload = {
        "kind": "s100_phase28_state_capture",
        "status": "started",
        "mode": mode,
        "context": CONTEXT,
        "started_utc": utc_now(),
    }

    runtime = None
    try:
        cfg, _, _, _ = phase28_gate()
        arm = selected_arm()
        trace = load_trace()
        tokens = trace["tokens"]

        if mode == "parent":
            runtime, graph, keep = make_synth(
                CONTEXT,
                cfg,
            )
        else:
            runtime, graph, keep = make_arm(
                CONTEXT,
                arm,
            )

        capture_info = graph.setup_graph()

        def run_once():
            runtime.reset()
            prefill_to(runtime, tokens, CONTEXT)
            graph.prepare_after_prefill()

            draft, expected = expected_for_block(
                tokens,
                CONTEXT,
            )
            ids = graph.launch(draft.tolist())
            if not np.array_equal(ids, expected):
                raise RuntimeError(
                    f"{mode} IDs mismatch: "
                    f"got={ids.tolist()} "
                    f"expected={expected.tolist()}"
                )
            return (
                np.asarray(ids, np.int32),
                expected,
                graph.v.logits,
            )

        ids, expected, logits = run_once()
        ids_repeat = None

        if mode == "candidate":
            ids_repeat, expected_repeat, _ = run_once()
            if not np.array_equal(
                ids_repeat,
                expected_repeat,
            ):
                raise RuntimeError(
                    "candidate deterministic replay diverged"
                )

            # Recreate the primary candidate state after the replay.
            ids, expected, logits = run_once()

        arrays = capture_arrays(
            runtime,
            logits,
            CONTEXT + 4,
            ids,
            ids_repeat=ids_repeat,
        )
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **arrays)

        wrapper = getattr(graph, "gmoe", None)
        payload.update(
            {
                "status": "measured",
                "selected_arm": arm.as_dict(),
                "capture_info": capture_info,
                "ids": ids.tolist(),
                "ids_repeat": (
                    None
                    if ids_repeat is None
                    else ids_repeat.tolist()
                ),
                "expected": expected.tolist(),
                "array_count": len(arrays),
                "mirror_bytes_removed": int(
                    getattr(
                        wrapper,
                        "freed_mirror_bytes",
                        0,
                    )
                ),
                "npz": str(out_npz),
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

    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_json, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "mode": mode,
                "selected_arm": payload.get(
                    "selected_arm"
                ),
                "ids": payload.get("ids"),
                "ids_repeat": payload.get(
                    "ids_repeat"
                ),
                "mirror_bytes_removed": payload.get(
                    "mirror_bytes_removed"
                ),
                "error": (
                    payload.get("error") or {}
                ).get("message"),
                "output": str(out_json),
            },
            indent=2,
        )
    )
    return payload


def compare() -> dict:
    parent_meta = json.loads(
        (
            RESULTS
            / "S100_PHASE28_STATE_PARENT.json"
        ).read_text(encoding="utf-8")
    )
    candidate_meta = json.loads(
        (
            RESULTS
            / "S100_PHASE28_STATE_CANDIDATE.json"
        ).read_text(encoding="utf-8")
    )

    if (
        parent_meta.get("status") != "measured"
        or candidate_meta.get("status") != "measured"
    ):
        raise RuntimeError(
            "Phase28 state captures are incomplete"
        )

    state, gates = compare_npz(
        RESULTS / "S100_PHASE28_STATE_PARENT.npz",
        RESULTS / "S100_PHASE28_STATE_CANDIDATE.npz",
    )

    result = {
        "kind": "s100_phase28_state_check",
        "status": "measured",
        "created_utc": utc_now(),
        "selected_arm": candidate_meta.get(
            "selected_arm"
        ),
        "state": state,
        "gates": gates,
        "parent_ids": parent_meta.get("ids"),
        "candidate_ids": candidate_meta.get("ids"),
        "candidate_repeat_ids": candidate_meta.get(
            "ids_repeat"
        ),
        "mirror_bytes_removed": candidate_meta.get(
            "mirror_bytes_removed"
        ),
        "SELECTED_STATE_GREEN": bool(
            all(gates.values())
        ),
    }

    path = RESULTS / "S100_PHASE28_STATE_CHECK.json"
    write_json_atomic(path, result, archive=True)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("parent", "candidate", "compare"),
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "compare":
        result = compare()
        return 0 if result.get(
            "SELECTED_STATE_GREEN"
        ) else 2

    result = capture(args.mode)
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
