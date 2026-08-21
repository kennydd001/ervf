"""Phase79 held-out DFlash hidden-to-target route projection."""
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
from s100_phase78_ornith_dflash_route_signal import _rows, _target_groups


RESULTS = REPO / "pro_research" / "results" / "s100_phase79"
PREREG = REPO / "pro_research" / "S100_PHASE79_ORNITH_DFLASH_ROUTE_PROJECTION_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase79_ornith_dflash_route_projection.py"
PHASE76 = REPO / "pro_research" / "results" / "s100_phase76" / "S100_PHASE76_ORNITH_FUTURE_ROUTER.json"
PHASE78 = REPO / "pro_research" / "results" / "s100_phase78" / "S100_PHASE78_ORNITH_DFLASH_ROUTE_SIGNAL.json"
TRAIN_EVENTS = 5
CORRECTORS = ("bias", "knn4", "ridge01", "ridge1")


def _fit_predict(name, train_x, test_x, residual):
    if name == "bias":
        return np.repeat(residual.mean(axis=0, keepdims=True), len(test_x), axis=0)
    train_norm = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1.0e-12)
    test_norm = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1.0e-12)
    if name == "knn4":
        k = min(4, len(train_x))
        similarity = test_norm @ train_norm.T
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        return np.stack([residual[row].mean(axis=0) for row in nearest])
    value = {"ridge01": 0.1, "ridge1": 1.0}[name]
    kernel = train_norm @ train_norm.T
    alpha = np.linalg.solve(
        kernel + value * np.eye(len(kernel), dtype=np.float32), residual
    )
    return (test_norm @ train_norm.T) @ alpha


def _aligned_events(trace, shift):
    batches = trace["target_batches"]
    target = _target_groups(trace["target"])[-len(batches):]
    draft = trace["draft"][-len(batches):]
    events = []
    for index, batch in enumerate(batches):
        length = len(batch)
        x = _rows(draft[index], 2048).astype(np.float32)[:length]
        layer_hidden = {
            layer: _rows(target[index]["hidden"][layer], 2048).astype(np.float32)
            for layer in range(40)
        }
        layer_routes = {
            layer: _rows(target[index]["routes"][layer], 8).astype(np.int64)
            for layer in range(40)
        }
        if shift:
            x = x[:-shift]
            layer_hidden = {layer: values[shift:] for layer, values in layer_hidden.items()}
            layer_routes = {layer: values[shift:] for layer, values in layer_routes.items()}
        events.append({"x": x, "hidden": layer_hidden, "routes": layer_routes})
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE79_ORNITH_DFLASH_ROUTE_PROJECTION.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase79_ornith_dflash_route_projection",
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
        routers, _norms = _load_router_weights(torch, safe_open, args.snapshot.resolve())
        torch.backends.cuda.matmul.allow_tf32 = False
        arms = {}
        for alignment, shift in (("same", 0), ("plus1", 1)):
            events = _aligned_events(trace, shift)
            train_x = np.concatenate([event["x"] for event in events[:TRAIN_EVENTS]])
            test_x_by_event = [event["x"] for event in events[TRAIN_EVENTS:]]
            test_x = np.concatenate(test_x_by_event)
            offsets = np.cumsum([0] + [len(value) for value in test_x_by_event])
            for corrector in CORRECTORS:
                assignment_hits = 0
                assignment_total = 0
                union_hits = 0
                union_total = 0
                for layer in range(40):
                    weight = routers[layer].float().cuda()
                    proxy_train = (
                        torch.from_numpy(train_x).cuda() @ weight.T
                    ).cpu().numpy()
                    proxy_test = (
                        torch.from_numpy(test_x).cuda() @ weight.T
                    ).cpu().numpy()
                    target_train_hidden = np.concatenate([
                        event["hidden"][layer] for event in events[:TRAIN_EVENTS]
                    ])
                    exact_train = (
                        torch.from_numpy(target_train_hidden).cuda() @ weight.T
                    ).cpu().numpy()
                    correction = _fit_predict(
                        corrector, train_x, test_x, exact_train - proxy_train
                    )
                    predicted_logits = proxy_test + correction
                    actual_test = np.concatenate([
                        event["routes"][layer] for event in events[TRAIN_EVENTS:]
                    ])
                    predicted = _top8(predicted_logits)
                    assignment_hits += sum(
                        len(set(predicted[token]) & set(actual_test[token]))
                        for token in range(len(actual_test))
                    )
                    assignment_total += len(actual_test) * 8
                    for event_index, event in enumerate(events[TRAIN_EVENTS:]):
                        begin, end = offsets[event_index], offsets[event_index + 1]
                        count = min(4, end - begin)
                        actual_union = set(
                            int(expert) for row in event["routes"][layer][:count] for expert in row
                        )
                        candidates = set(_candidate_union(
                            predicted_logits[begin:begin + count], 32
                        ))
                        union_hits += len(actual_union & candidates)
                        union_total += len(actual_union)
                name = f"{alignment}_{corrector}"
                arms[name] = {
                    "alignment": alignment,
                    "shift": shift,
                    "corrector": corrector,
                    "train_events": TRAIN_EVENTS,
                    "test_events": len(events) - TRAIN_EVENTS,
                    "assignment_hits": assignment_hits,
                    "assignment_total": assignment_total,
                    "assignment_recall": assignment_hits / assignment_total,
                    "h4_union_hits": union_hits,
                    "h4_union_total": union_total,
                    "h4_unique_route_recall": union_hits / union_total,
                }
        primary = [row for row in arms.values() if row["alignment"] == "plus1"]
        winner = max(primary, key=lambda row: row["h4_unique_route_recall"])
        phase76 = json.loads(PHASE76.read_text("utf-8"))
        phase78 = json.loads(PHASE78.read_text("utf-8"))
        upstream = (
            phase76["gates"]["P76_G1_trace_and_router_parity"]
            and phase78["gates"]["P78_G1_callback_event_alignment"]
        )
        gates = {
            "P79_G1_upstream_alignment_and_router": upstream,
            "P79_G2_projected_route_signal": (
                winner["assignment_recall"] >= 0.80
                or winner["h4_unique_route_recall"] >= 0.95
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(args.trace.resolve()),
                "snapshot": str(args.snapshot.resolve()),
                "train_events": TRAIN_EVENTS,
                "test_events": len(trace["target_batches"]) - TRAIN_EVENTS,
            },
            "arms": arms,
            "winner": next(name for name, row in arms.items() if row is winner),
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
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE76, PHASE78, args.trace))
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "winner": payload.get("winner"),
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
