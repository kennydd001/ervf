from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

CURRENT_REDUCE = {38, 10, 40, 20, 43, 13}
CURRENT_BOOST = {1, 3, 51, 6}
UP_CODE = 2_494_464
UP_SCALE = 311_808
DOWN_PLANE = 311_808
SLOT_BYTES = UP_CODE + UP_SCALE + DOWN_PLANE
PCIE_GBS = 26.1686


def cmap_current(layers):
    return {
        int(layer): (
            52 if int(layer) in CURRENT_REDUCE
            else 102 if int(layer) in CURRENT_BOOST
            else 72
        )
        for layer in layers
    }


def lru_layer(ids, counted, sessions, capacity, session_filter=None):
    hits = misses = 0
    current_session = None
    cache = collections.OrderedDict()

    for token in range(len(ids)):
        sid = int(sessions[token])
        if session_filter is not None and not session_filter(sid):
            continue
        if current_session != sid:
            current_session = sid
            cache = collections.OrderedDict()

        for raw_expert in ids[token]:
            expert = int(raw_expert)
            hit = expert in cache
            if counted[token]:
                hits += int(hit)
                misses += int(not hit)
            if hit:
                cache.move_to_end(expert)
            else:
                cache[expert] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return hits, misses


def static_layer(ids, counted, sessions, capacity, train_filter, test_filter):
    frequency = collections.Counter()
    for token in range(len(ids)):
        if train_filter(int(sessions[token])):
            for expert in ids[token]:
                frequency[int(expert)] += 1

    hot = {expert for expert, _ in frequency.most_common(capacity)}
    hits = misses = 0
    for token in range(len(ids)):
        if test_filter(int(sessions[token])) and counted[token]:
            for expert in ids[token]:
                hit = int(expert) in hot
                hits += int(hit)
                misses += int(not hit)
    return hits, misses


def belady_layer(ids, counted, sessions, capacity, session_filter):
    hits = misses = 0
    session_ids = sorted({
        int(value) for value in sessions if session_filter(int(value))
    })

    for sid in session_ids:
        token_indices = np.flatnonzero(sessions == sid)
        flat_experts = []
        flat_counted = []
        for token in token_indices:
            for expert in ids[token]:
                flat_experts.append(int(expert))
                flat_counted.append(bool(counted[token]))

        future = collections.defaultdict(collections.deque)
        for position, expert in enumerate(flat_experts):
            future[expert].append(position)

        cache = set()
        for position, (expert, is_counted) in enumerate(
            zip(flat_experts, flat_counted)
        ):
            queue = future[expert]
            if queue and queue[0] == position:
                queue.popleft()

            hit = expert in cache
            if is_counted:
                hits += int(hit)
                misses += int(not hit)
            if hit:
                continue

            if len(cache) >= capacity:
                victim = max(
                    cache,
                    key=lambda item: (
                        future[item][0] if future[item] else 10**18
                    ),
                )
                cache.remove(victim)
            cache.add(expert)
    return hits, misses


def optimize_caps(layers, train_miss, budget, capacities):
    """Exact multiple-choice knapsack over one capacity per MoE layer."""
    dp = {0: (0, [])}  # used slots -> (miss cost, chosen capacities)

    for layer in layers:
        next_dp = {}
        for used, (cost, chosen) in dp.items():
            for capacity in capacities:
                new_used = used + capacity
                if new_used > budget:
                    continue
                new_cost = cost + int(train_miss[int(layer)][capacity])
                previous = next_dp.get(new_used)
                if previous is None or new_cost < previous[0]:
                    next_dp[new_used] = (new_cost, chosen + [capacity])
        if not next_dp:
            raise RuntimeError(
                f"No feasible cache allocation after layer {layer}; "
                f"budget={budget}"
            )
        dp = next_dp

    used, (cost, chosen) = min(
        dp.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    if len(chosen) != len(layers):
        raise RuntimeError(
            f"DP returned {len(chosen)} capacities for {len(layers)} layers"
        )

    mapping = {
        int(layer): int(capacity)
        for layer, capacity in zip(layers, chosen)
    }
    if sum(mapping.values()) != used:
        raise RuntimeError("DP capacity sum does not match used-slot record")
    return mapping, int(used), int(cost)


def eval_map(ids, counted, sessions, layers, capmap, session_filter):
    hits = misses = 0
    selected = np.asarray(
        [session_filter(int(value)) for value in sessions],
        dtype=bool,
    )
    counted_tokens = int(counted[selected].sum())

    for layer_index, layer in enumerate(layers):
        layer_hits, layer_misses = lru_layer(
            ids[:, layer_index, :],
            counted,
            sessions,
            int(capmap[int(layer)]),
            session_filter,
        )
        hits += layer_hits
        misses += layer_misses

    total = hits + misses
    denominator = counted_tokens * len(layers)
    return {
        "hits": int(hits),
        "misses": int(misses),
        "miss_fraction": misses / total if total else None,
        "misses_per_layer_token": (
            misses / denominator if denominator else None
        ),
        "counted_tokens": counted_tokens,
    }


def markov_prefetch(
    ids,
    counted,
    sessions,
    layers,
    capmap,
    train_filter,
    test_filter,
    budget,
):
    layer_count = len(layers)
    expert_count = 128
    transitions = np.ones(
        (layer_count, expert_count, expert_count),
        dtype=np.float64,
    ) * 0.01

    previous = {}
    for token in range(len(ids)):
        sid = int(sessions[token])
        if not train_filter(sid):
            continue
        if sid in previous:
            for layer_index in range(layer_count):
                for source in previous[sid][layer_index]:
                    for target in ids[token, layer_index]:
                        transitions[
                            layer_index, int(source), int(target)
                        ] += 1
        previous[sid] = ids[token].copy()

    caches = [collections.OrderedDict() for _ in range(layer_count)]
    prefetched = [set() for _ in range(layer_count)]
    current_session = None
    hits = prefetch_hits = misses = 0
    prefetches = prefetch_used = 0

    for token in range(len(ids)):
        sid = int(sessions[token])
        if not test_filter(sid):
            continue
        if current_session != sid:
            caches = [
                collections.OrderedDict() for _ in range(layer_count)
            ]
            prefetched = [set() for _ in range(layer_count)]
            current_session = sid

        for layer_index, layer in enumerate(layers):
            cache = caches[layer_index]
            for raw_expert in ids[token, layer_index]:
                expert = int(raw_expert)
                main_hit = expert in cache
                prefetched_hit = expert in prefetched[layer_index]

                if counted[token]:
                    if main_hit:
                        hits += 1
                    elif prefetched_hit:
                        prefetch_hits += 1
                    else:
                        misses += 1

                if main_hit:
                    cache.move_to_end(expert)
                else:
                    cache[expert] = None
                    if len(cache) > int(capmap[int(layer)]):
                        cache.popitem(last=False)

                if prefetched_hit and counted[token]:
                    prefetch_used += 1

        candidates = []
        for layer_index in range(layer_count):
            score = transitions[
                layer_index, ids[token, layer_index].astype(int)
            ].sum(axis=0)
            denominator = float(score.sum())
            for expert in np.argsort(score)[-3:][::-1]:
                expert = int(expert)
                if expert not in caches[layer_index]:
                    candidates.append(
                        (float(score[expert] / denominator), layer_index, expert)
                    )

        candidates.sort(reverse=True)
        next_prefetched = [set() for _ in range(layer_count)]
        next_is_counted = (
            token + 1 < len(ids)
            and int(sessions[token + 1]) == sid
            and bool(counted[token + 1])
        )
        for _, layer_index, expert in candidates:
            if sum(len(values) for values in next_prefetched) >= budget:
                break
            if expert not in next_prefetched[layer_index]:
                next_prefetched[layer_index].add(expert)
                prefetches += int(next_is_counted)
        prefetched = next_prefetched

    total = hits + prefetch_hits + misses
    counted_test_tokens = int(
        counted[
            np.asarray(
                [test_filter(int(value)) for value in sessions],
                dtype=bool,
            )
        ].sum()
    )
    return {
        "budget_records_per_token": budget,
        "demand_miss_fraction": misses / total if total else None,
        "main_hits": hits,
        "prefetch_hits": prefetch_hits,
        "demand_misses": misses,
        "prefetches": prefetches,
        "prefetch_used": prefetch_used,
        "prefetch_precision": (
            prefetch_used / prefetches if prefetches else 0.0
        ),
        "bytes_prefetched_per_counted_token": (
            prefetches * SLOT_BYTES / max(1, counted_test_tokens)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.trace) as trace:
        ids = trace["ids"].astype(np.int16)
        counted = trace["counted"].astype(bool)
        sessions = trace["session"].astype(np.int16)
        layers = [int(value) for value in trace["layers"]]
        need = trace["need"].astype(np.int8)

    train = lambda sid: sid % 2 == 0
    test = lambda sid: sid % 2 == 1
    current = cmap_current(layers)
    capacities = list(range(32, 129, 2))

    actual_need = float(need[counted].sum() / need[counted].size)
    current_all = eval_map(
        ids, counted, sessions, layers, current, lambda _: True
    )
    simulation_error = abs(
        float(current_all["miss_fraction"]) - actual_need
    )
    simulation_gate = simulation_error <= 0.015

    train_miss = {}
    curves = {}
    for layer_index, layer in enumerate(layers):
        train_miss[layer] = {}
        curves[str(layer)] = {}
        for capacity in capacities:
            train_hits, train_misses = lru_layer(
                ids[:, layer_index, :],
                counted,
                sessions,
                capacity,
                train,
            )
            test_hits, test_misses = lru_layer(
                ids[:, layer_index, :],
                counted,
                sessions,
                capacity,
                test,
            )
            train_miss[layer][capacity] = train_misses
            curves[str(layer)][str(capacity)] = {
                "train_hits": train_hits,
                "train_misses": train_misses,
                "test_hits": test_hits,
                "test_misses": test_misses,
            }

    budgets = [1656, 1784, 1912, 2035]
    profiles = {
        "current": {str(key): value for key, value in current.items()}
    }
    optimized_rows = {}

    for budget in budgets:
        mapping, used, cost = optimize_caps(
            layers, train_miss, budget, capacities
        )
        evaluation = eval_map(
            ids, counted, sessions, layers, mapping, test
        )
        name = (
            "budget_neutral"
            if budget == 1656
            else f"plus_{budget - 1656}"
        )
        profiles[name] = {
            str(key): value for key, value in mapping.items()
        }
        optimized_rows[name] = {
            "slot_budget": budget,
            "slots_used": used,
            "train_misses": cost,
            "test": evaluation,
            "extra_vram_mib_estimate": (
                max(0, used - 1656) * SLOT_BYTES / 1024**2
            ),
        }

    static_hits = static_misses = 0
    belady_hits = belady_misses = 0
    for layer_index, layer in enumerate(layers):
        hits, misses = static_layer(
            ids[:, layer_index, :],
            counted,
            sessions,
            current[layer],
            train,
            test,
        )
        static_hits += hits
        static_misses += misses

        hits, misses = belady_layer(
            ids[:, layer_index, :],
            counted,
            sessions,
            current[layer],
            test,
        )
        belady_hits += hits
        belady_misses += misses

    prefetch = [
        markov_prefetch(
            ids,
            counted,
            sessions,
            layers,
            current,
            train,
            test,
            budget,
        )
        for budget in (4, 8, 12)
    ]
    test_current = eval_map(
        ids, counted, sessions, layers, current, test
    )

    profile_invariants = {}
    for name, mapping_raw in profiles.items():
        mapping = {int(key): int(value) for key, value in mapping_raw.items()}
        profile_invariants[name] = {
            "layer_count": len(mapping),
            "slots": sum(mapping.values()),
            "all_layers_present": set(mapping) == set(layers),
            "all_caps_even_32_to_128": all(
                32 <= value <= 128 and value % 2 == 0
                for value in mapping.values()
            ),
        }

    output = {
        "kind": "s100_phase9_cache_oracle",
        "status": "measured",
        "simulation_gate": simulation_gate,
        "simulation_error_fraction": simulation_error,
        "measured_miss_fraction": actual_need,
        "simulated_current_all": current_all,
        "test_current": test_current,
        "static_train_frequency_test": {
            "miss_fraction": (
                static_misses / (static_hits + static_misses)
            ),
        },
        "belady_current_map_test": {
            "miss_fraction": (
                belady_misses / (belady_hits + belady_misses)
            ),
        },
        "optimized_profiles": optimized_rows,
        "prefetch": prefetch,
        "slot_bytes": SLOT_BYTES,
        "pcie_gbs_anchor": PCIE_GBS,
        "theoretical_current_up_fetch_serial_ms": (
            float(current_all["misses_per_layer_token"])
            * len(layers)
            * (UP_CODE + UP_SCALE)
            / (PCIE_GBS * 1e9)
            * 1e3
        ),
        "profile_invariants": profile_invariants,
        "profiles_path": "S100_PHASE9_CAPACITY_PROFILES.json",
        "curves": curves,
    }

    oracle_path = output_dir / "S100_PHASE9_CACHE_ORACLE.json"
    profiles_path = output_dir / "S100_PHASE9_CAPACITY_PROFILES.json"
    oracle_path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    profiles_path.write_text(
        json.dumps(
            {"profiles": profiles, "oracle": optimized_rows},
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if simulation_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
