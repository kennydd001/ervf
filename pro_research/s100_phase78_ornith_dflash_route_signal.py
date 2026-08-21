"""Phase78 route signal in real aligned DFlash hidden states."""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase76_ornith_future_router import _candidate_union, _load_router_weights, _top8


RESULTS = REPO / "pro_research" / "results" / "s100_phase78"
PREREG = REPO / "pro_research" / "S100_PHASE78_ORNITH_DFLASH_ROUTE_SIGNAL_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase78_ornith_dflash_route_signal.py"
PHASE76 = REPO / "pro_research" / "results" / "s100_phase76" / "S100_PHASE76_ORNITH_FUTURE_ROUTER.json"


def _target_groups(items):
    if len(items) % 80:
        raise ValueError("target callback tensor count is not divisible by 80")
    groups = []
    for begin in range(0, len(items), 80):
        block = items[begin:begin + 80]
        routes = {}
        hidden = {}
        for tensor in block:
            if tensor["name"].startswith("ffn_moe_topk-"):
                routes[int(tensor["name"].rsplit("-", 1)[1])] = tensor
            elif tensor["name"].startswith("attn_post_norm-"):
                hidden[int(tensor["name"].rsplit("-", 1)[1])] = tensor
        if sorted(routes) != list(range(40)) or sorted(hidden) != list(range(40)):
            raise ValueError("target callback group lacks 40 route/hidden pairs")
        groups.append({"routes": routes, "hidden": hidden})
    return groups


def _rows(tensor, width: int):
    count = int(tensor["shape"][1])
    return np.asarray(tensor["values"]).reshape(count, width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE78_ORNITH_DFLASH_ROUTE_SIGNAL.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase78_ornith_dflash_route_signal",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        import torch
        from safetensors import safe_open

        if args.trace.suffix == ".gz":
            with gzip.open(args.trace, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
        else:
            trace = json.loads(args.trace.read_text("utf-8"))
        target_batches = trace["target_batches"]
        target = _target_groups(trace["target"])[-len(target_batches):]
        draft = trace["draft"][-len(target_batches):]
        target_lengths = [len(row) for row in target_batches]
        target_tensor_lengths = [int(group["routes"][0]["shape"][1]) for group in target]
        draft_lengths = [int(tensor["shape"][1]) for tensor in draft]
        alignment_ok = (
            target_lengths == target_tensor_lengths
            and all(draft_length >= target_length for draft_length, target_length in zip(
                draft_lengths, target_lengths
            ))
        )
        routers, _norms = _load_router_weights(
            torch, safe_open, args.snapshot.resolve()
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        logits = {layer: [] for layer in range(40)}
        for event, tensor in enumerate(draft):
            hidden = torch.from_numpy(_rows(tensor, 2048).astype(np.float32)).cuda()
            for layer in range(40):
                weight = routers[layer].float().cuda()
                logits[layer].append((hidden @ weight.T).cpu().numpy())

        arms = {}
        for name, shift in (("same", 0), ("plus1", 1)):
            assignment_hits = 0
            assignment_total = 0
            h4 = {32: {"hits": 0, "total": 0}, 64: {"hits": 0, "total": 0}}
            event_rows = []
            for event, group in enumerate(target):
                available = target_lengths[event] - shift
                event_hits = 0
                event_total = 0
                for layer in range(40):
                    predicted = _top8(logits[layer][event][:available])
                    actual = _rows(group["routes"][layer], 8).astype(np.int64)[shift:]
                    hits = sum(
                        len(set(predicted[token]) & set(actual[token]))
                        for token in range(available)
                    )
                    event_hits += hits
                    event_total += available * 8
                    h4_count = min(4, available)
                    actual_union = set(int(expert) for row in actual[:h4_count] for expert in row)
                    for budget in (32, 64):
                        candidates = set(_candidate_union(logits[layer][event][:h4_count], budget))
                        h4[budget]["hits"] += len(actual_union & candidates)
                        h4[budget]["total"] += len(actual_union)
                assignment_hits += event_hits
                assignment_total += event_total
                event_rows.append({
                    "event": event,
                    "batch_tokens": target_batches[event],
                    "positions": available,
                    "assignment_recall": event_hits / event_total,
                })
            arms[name] = {
                "shift": shift,
                "assignment_hits": assignment_hits,
                "assignment_total": assignment_total,
                "assignment_recall": assignment_hits / assignment_total,
                "h4_unique_route_recall": {
                    str(budget): row["hits"] / row["total"] for budget, row in h4.items()
                },
                "h4_counts": h4,
                "events": event_rows,
            }
        phase76 = json.loads(PHASE76.read_text("utf-8"))
        exact_router_ok = phase76["gates"]["P76_G1_trace_and_router_parity"]
        primary = arms["plus1"]
        gates = {
            "P78_G1_callback_event_alignment": alignment_ok,
            "P78_G2_exact_target_router_upstream": exact_router_ok,
            "P78_G3_dflash_route_signal": (
                primary["assignment_recall"] >= 0.80
                or primary["h4_unique_route_recall"]["32"] >= 0.95
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(args.trace.resolve()),
                "snapshot": str(args.snapshot.resolve()),
                "target_batches": len(target_batches),
            },
            "alignment": {
                "metadata_lengths": target_lengths,
                "target_tensor_lengths": target_tensor_lengths,
                "draft_tensor_lengths": draft_lengths,
                "exact": alignment_ok,
            },
            "arms": arms,
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE76, args.trace))
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "alignment": payload.get("alignment"),
        "arms": {
            name: {
                "assignment_recall": row["assignment_recall"],
                "h4_unique_route_recall": row["h4_unique_route_recall"],
            }
            for name, row in (payload.get("arms") or {}).items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
