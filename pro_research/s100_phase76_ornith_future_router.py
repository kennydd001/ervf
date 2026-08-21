"""Phase76 destination-router-on-earlier-hidden-state audit."""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase76"
PREREG = REPO / "pro_research" / "S100_PHASE76_ORNITH_FUTURE_ROUTER_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase76_ornith_future_router.py"
RUNNER = REPO / "pro_research" / "llama_ornith_trace.cpp"
PHASE70_TRACE = REPO / "pro_research" / "results" / "s100_phase70" / "ornith_128_trace.json"
PHASE71 = REPO / "pro_research" / "results" / "s100_phase71" / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE.json"
PHASE73 = REPO / "pro_research" / "results" / "s100_phase73" / "S100_PHASE73_ORNITH_SEGMENTED_REALCOMPUTE.json"
FLOOR_MS = 60.095487602
BOUNDARY_MS = 4000.0 / 65.0
WARMUP_TOKENS = 32
LEADS = (1, 2, 4)
BUDGETS = (8, 16, 24, 32)
VARIANTS = ("direct", "direct_bias", "normswap", "normswap_bias")


def _load_hidden_trace(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text("utf-8"))
    tokens = tuple(int(value) for value in payload["tokens"])
    routes = {}
    hidden = {}
    for tensor in payload["tensors"]:
        name = tensor["name"]
        if name.startswith("ffn_moe_topk-"):
            layer = int(name.rsplit("-", 1)[1])
            values = tensor["values"]
            routes[layer] = tuple(
                tuple(int(value) for value in values[token * 8:(token + 1) * 8])
                for token in range(len(tokens))
            )
        elif name.startswith("attn_post_norm-"):
            layer = int(name.rsplit("-", 1)[1])
            hidden[layer] = np.asarray(tensor["values"], dtype=np.float32).reshape(
                len(tokens), 2048
            )
    if sorted(routes) != list(range(40)) or sorted(hidden) != list(range(40)):
        raise ValueError("hidden trace must contain routes and attn_post_norm for layers 0..39")
    return payload, tokens, routes, hidden


def _load_router_weights(torch, safe_open, snapshot: Path):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
    weight_map = index["weight_map"]
    required = []
    for layer in range(40):
        required.extend((
            f"model.layers.{layer}.mlp.gate.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
        ))
    by_file: dict[str, list[str]] = {}
    for name in required:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for filename, names in by_file.items():
        with safe_open(str(snapshot / filename), framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    routers = tuple(tensors[f"model.layers.{layer}.mlp.gate.weight"] for layer in range(40))
    norms = tuple(
        tensors[f"model.layers.{layer}.post_attention_layernorm.weight"]
        for layer in range(40)
    )
    return routers, norms


def _top8(logits: np.ndarray) -> np.ndarray:
    return np.argsort(-logits, axis=1, kind="stable")[:, :8]


def _candidate_union(logits: np.ndarray, budget: int) -> tuple[int, ...]:
    per_token = _top8(logits)
    union = list(dict.fromkeys(int(expert) for row in per_token for expert in row))
    if len(union) <= budget:
        return tuple(union)
    maximum = logits[:, union].max(axis=0)
    order = np.argsort(-maximum, kind="stable")[:budget]
    return tuple(union[int(index)] for index in order)


def _frequency_candidates(routes, layer: int, begin: int, budget: int):
    counts = Counter(expert for row in routes[layer][:begin] for expert in row)
    last = {}
    for token, row in enumerate(routes[layer][:begin]):
        for expert in row:
            last[expert] = token
    ranked = sorted(counts, key=lambda expert: (-counts[expert], -last[expert], expert))
    ranked.extend(expert for expert in range(256) if expert not in set(ranked))
    return tuple(ranked[:budget])


def _cache_misses(routes):
    import sys

    source = REPO / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from moe_lab.ornith.rolling_prefetch import (
        RollingPrefetchController,
        build_execution_layer_plan,
    )

    controller = RollingPrefetchController()
    controller.reset_request("phase76:authoritative")
    result = {}
    for begin in range(0, 64, 4):
        actual = tuple(routes[layer][begin:begin + 4] for layer in range(40))
        snapshots = controller.cache_snapshot()
        result[begin] = tuple(
            build_execution_layer_plan(actual[layer], snapshots[layer], (), layer=layer).uncovered_experts
            for layer in range(40)
        )
        block = controller.prepare_block(actual)
        controller.adjudicate(block.block_id, actual)
    return result


def _evaluate_arm(name, lead, budget, variant, proxies, exact_logits, routes, misses,
                  serial_group_ms, overlap_tail_ms):
    totals = Counter()
    block_rows = []
    use_bias = variant.endswith("_bias")
    base = variant.removesuffix("_bias")
    proxy = proxies[(lead, base)]
    for begin in range(WARMUP_TOKENS, 64, 4):
        block = Counter()
        for destination in range(40):
            if destination < lead:
                candidates = _frequency_candidates(routes, destination, begin, budget)
            else:
                values = proxy[destination][begin:begin + 4].copy()
                if use_bias:
                    correction = (
                        exact_logits[destination][:begin]
                        - proxy[destination][:begin]
                    ).mean(axis=0)
                    values += correction[None, :]
                candidates = _candidate_union(values, budget)
            true_misses = set(misses[begin][destination])
            candidate_set = set(candidates)
            block["candidates"] += len(candidates)
            block["hits"] += len(true_misses & candidate_set)
            block["uncovered"] += len(true_misses - candidate_set)
            block["false"] += len(candidate_set - true_misses)
        totals.update(block)
        block_rows.append({"begin_token": begin, **dict(block)})
    actual_misses = totals["hits"] + totals["uncovered"]
    recall = totals["hits"] / actual_misses
    precision = totals["hits"] / totals["candidates"]
    mean_uncovered = totals["uncovered"] / len(block_rows)
    projected_ms = FLOOR_MS + overlap_tail_ms + mean_uncovered * serial_group_ms
    return {
        "name": name,
        "lead": lead,
        "budget": budget,
        "variant": variant,
        "evaluation_blocks": len(block_rows),
        "totals": dict(totals),
        "unique_miss_recall": recall,
        "candidate_precision": precision,
        "mean_uncovered_groups_h4": mean_uncovered,
        "optimistic_projected_ms_h4": projected_ms,
        "optimistic_projected_tok_s": 4000.0 / projected_ms,
        "blocks": block_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE76_ORNITH_FUTURE_ROUTER.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase76_ornith_future_router",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        import torch
        from safetensors import safe_open

        trace_path = args.trace.resolve()
        snapshot = args.snapshot.resolve()
        _raw, tokens, routes, hidden = _load_hidden_trace(trace_path)
        phase71 = json.loads(PHASE71.read_text("utf-8"))
        phase73 = json.loads(PHASE73.read_text("utf-8"))
        phase70_raw = json.loads(PHASE70_TRACE.read_text("utf-8"))
        token_match = tokens == tuple(phase70_raw["tokens"][:64])
        route_match = all(
            routes[layer] == tuple(
                tuple(int(value) for value in next(
                    tensor["values"] for tensor in phase70_raw["tensors"]
                    if tensor["name"] == f"ffn_moe_topk-{layer}"
                )[token * 8:(token + 1) * 8])
                for token in range(64)
            )
            for layer in range(40)
        )
        routers, norms = _load_router_weights(torch, safe_open, snapshot)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        exact_logits = {}
        proxies = {(lead, base): {} for lead in LEADS for base in ("direct", "normswap")}
        for destination in range(40):
            weight = routers[destination].float().to(device)
            exact_input = torch.from_numpy(hidden[destination]).to(device)
            exact_logits[destination] = (exact_input @ weight.T).cpu().numpy()
            for lead in LEADS:
                if destination < lead:
                    continue
                source = destination - lead
                direct = torch.from_numpy(hidden[source]).to(device)
                proxies[(lead, "direct")][destination] = (direct @ weight.T).cpu().numpy()
                source_norm = norms[source].float().to(device)
                destination_norm = norms[destination].float().to(device)
                swapped = direct / source_norm.clamp_min(1.0e-6) * destination_norm
                proxies[(lead, "normswap")][destination] = (swapped @ weight.T).cpu().numpy()
            del weight, exact_input
        parity_hits = 0
        parity_total = 40 * 64 * 8
        parity_by_layer = {}
        for layer in range(40):
            predicted = _top8(exact_logits[layer])
            hits = sum(
                len(set(predicted[token]) & set(routes[layer][token]))
                for token in range(64)
            )
            parity_hits += hits
            parity_by_layer[str(layer)] = hits / (64 * 8)
        misses = _cache_misses(routes)
        lru71 = phase71["records"]["lru52"]
        serial_group_ms = (
            lru71["summary"]["serial_increment_ms_h4"]
            / lru71["trace"]["mean_groups_per_h4"]
        )
        overlap_tail_ms = phase73["records"]["lru52"]["selected"]["exposed_tail_ms_h4"]
        arms = {}
        for lead in LEADS:
            for budget in BUDGETS:
                for variant in VARIANTS:
                    name = f"lead{lead}_b{budget}_{variant}"
                    arms[name] = _evaluate_arm(
                        name, lead, budget, variant, proxies, exact_logits, routes,
                        misses, serial_group_ms, overlap_tail_ms,
                    )
        physical = [row for row in arms.values() if row["lead"] >= 2]
        winner = max(
            physical,
            key=lambda row: (row["unique_miss_recall"], -row["totals"]["candidates"]),
        )
        oracle_uncovered = sum(
            len(set(misses[begin][layer]) - {
                expert for row in routes[layer][begin:begin + 4] for expert in row
            })
            for begin in range(WARMUP_TOKENS, 64, 4) for layer in range(40)
        )
        parity = parity_hits / parity_total
        gates = {
            "P76_G1_trace_and_router_parity": token_match and route_match and parity >= 0.999,
            "P76_G2_oracle_zero_uncovered": oracle_uncovered == 0,
            "P76_G3_physical_lead_recall_ge_95pct": winner["unique_miss_recall"] >= 0.95,
            "P76_G4_projected_boundary_le_65": winner["optimistic_projected_ms_h4"] <= BOUNDARY_MS,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(trace_path),
                "snapshot": str(snapshot),
                "tokens": len(tokens),
                "warmup_tokens": WARMUP_TOKENS,
                "serial_group_ms": serial_group_ms,
                "phase73_overlap_tail_ms": overlap_tail_ms,
                "all_hot_floor_ms_h4": FLOOR_MS,
                "boundary_ms_h4": BOUNDARY_MS,
            },
            "trace_contract": {"token_match": token_match, "route_match": route_match},
            "router_parity": {
                "assignment_recall": parity,
                "hits": parity_hits,
                "total": parity_total,
                "by_layer": parity_by_layer,
            },
            "oracle_uncovered": oracle_uncovered,
            "arms": arms,
            "winner": winner["name"],
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
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, RUNNER, args.trace, PHASE70_TRACE, PHASE71, PHASE73))
    write_json_atomic(out, payload, archive=True)
    ranked = sorted(
        (payload.get("arms") or {}).values(),
        key=lambda row: row["unique_miss_recall"],
        reverse=True,
    )[:12]
    print(json.dumps({
        "status": payload.get("status"),
        "trace_contract": payload.get("trace_contract"),
        "router_parity": payload.get("router_parity"),
        "winner": payload.get("winner"),
        "top_arms": [{
            "name": row["name"],
            "recall": row["unique_miss_recall"],
            "precision": row["candidate_precision"],
            "uncovered_h4": row["mean_uncovered_groups_h4"],
            "projected_tok_s": row["optimistic_projected_tok_s"],
        } for row in ranked],
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
