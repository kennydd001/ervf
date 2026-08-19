from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from common import REPO, write_json_atomic, utc_now
from diag_component_marginals_graph import (
    _prefill, _reset_exact_state
)
from s100_phase10a_runtime import build
from s100_phase9_trace import load_prompts
from s100_lightning16_common import (
    RESULTS, assert_lightning, ensure_results,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    args = parser.parse_args()
    ensure_results()
    output_npz = RESULTS / "S100_LIGHTNING16_ROUTE_TRACE.npz"
    output_json = RESULTS / "S100_LIGHTNING16_ROUTE_TRACE.json"
    payload = {
        "kind": "s100_lightning16_route_trace",
        "status": "started",
        "target_tokens": args.tokens,
        "started_utc": utc_now(),
    }
    bundle = None
    try:
        import cupy as cp

        ident = assert_lightning()
        prompts = load_prompts(REPO)
        bundle = build()
        rt = bundle.rt
        layers = [int(x) for x in rt.moe_layers]
        top_k = int(rt.top_k)

        ids_rows = []
        need_rows = []
        counted = []
        sessions = []
        prompt_indices = []
        token_in_session = []
        done = 0
        session_id = 0

        while done < args.tokens:
            prompt = prompts[session_id % len(prompts)]
            _reset_exact_state(rt)
            _prefill(rt, prompt["prompt_ids"])
            measured_this_session = min(
                224, args.tokens - done
            )
            total_steps = 32 + measured_this_session

            for step in range(total_steps):
                rt.step_graph(None)
                rt._graph_stream.synchronize()
                ids = cp.asnumpy(cp.stack([
                    rt._dev_cache[layer]["ids"][:top_k]
                    for layer in layers
                ])).astype(np.int16)
                need = cp.asnumpy(cp.stack([
                    rt._dev_cache[layer]["need"][:top_k]
                    for layer in layers
                ])).astype(np.int8)
                is_counted = step >= 32
                ids_rows.append(ids.copy())
                need_rows.append(need.copy())
                counted.append(is_counted)
                sessions.append(session_id)
                prompt_indices.append(prompt["index"])
                token_in_session.append(step)
                if is_counted:
                    done += 1

            session_id += 1
            if session_id % 8 == 0:
                print(
                    f"Lightning route trace {done}/{args.tokens}",
                    flush=True,
                )

        ids_array = np.stack(ids_rows)
        need_array = np.stack(need_rows)
        counted_array = np.asarray(counted, bool)
        np.savez_compressed(
            output_npz,
            ids=ids_array,
            need=need_array,
            counted=counted_array,
            session=np.asarray(sessions, np.int16),
            prompt_index=np.asarray(prompt_indices, np.int16),
            token_in_session=np.asarray(token_in_session, np.int16),
            layers=np.asarray(layers, np.int16),
        )
        measured_need = need_array[counted_array]
        payload.update({
            "status": "measured",
            "identity": ident,
            "measured_tokens": int(counted_array.sum()),
            "all_steps": int(len(counted_array)),
            "sessions": session_id,
            "prompt_pool": len(prompts),
            "shape": {
                "hidden": int(rt.hidden),
                "moe_inter": int(rt.moe_inter),
                "n_experts": int(rt.n_experts),
                "top_k": top_k,
                "moe_layers": layers,
            },
            "measured_route_slot_miss_fraction": float(
                measured_need.sum() / measured_need.size
            ),
            "npz": str(output_npz.relative_to(REPO)),
            "npz_bytes": output_npz.stat().st_size,
            "completed_utc": utc_now(),
        })
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
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(output_json, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "measured_tokens": payload.get("measured_tokens"),
        "miss_fraction": payload.get(
            "measured_route_slot_miss_fraction"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output_json),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
