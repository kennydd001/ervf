from __future__ import annotations

import pytest

from moe_lab.ornith.trace_analysis import (
    parse_llama_trace,
    replay_expert_cache,
    summarize_h4_miss_groups,
)


def _payload():
    tensors = []
    for layer in range(40):
        routes = list(range(8)) + list(range(4, 12))
        weights = [0.125] * 16
        tensors.extend([
            {"name": f"ffn_moe_topk-{layer}", "shape": [8, 2, 1, 1], "values": routes},
            {"name": f"ffn_moe_weights_norm-{layer}", "shape": [8, 2, 1, 1], "values": weights},
        ])
    tensors.append({
        "name": "result_norm", "shape": [2048, 2, 1, 1], "values": [0.0] * 4096,
    })
    return {"tokens": [10, 11], "tensors": tensors}


def test_parse_llama_trace_reconstructs_token_major_rows():
    trace = parse_llama_trace(_payload())
    assert trace.routes[0][0] == tuple(range(8))
    assert trace.routes[0][1] == tuple(range(4, 12))
    assert len(trace.result_norm) == 2
    assert len(trace.result_norm[0]) == 2048


def test_lru_lookups_are_atomic_within_token():
    replay = replay_expert_cache((tuple(range(8)), tuple(range(4, 12))), slots=8, policy="lru")
    assert replay["per_token"][0]["hits"] == []
    assert replay["per_token"][1]["hits"] == [4, 5, 6, 7]
    assert replay["hits"] == 4
    assert replay["misses"] == 12


def test_belady_is_never_worse_on_known_fixture():
    routes = (
        tuple(range(8)),
        tuple(range(4, 12)),
        (0, 1, 2, 3, 8, 9, 10, 11),
        tuple(range(8)),
    )
    lru = replay_expert_cache(routes, slots=8, policy="lru")
    belady = replay_expert_cache(routes, slots=8, policy="belady")
    assert belady["hits"] >= lru["hits"]


def test_parser_rejects_missing_layers_and_bad_routes():
    payload = _payload()
    payload["tensors"] = payload["tensors"][2:]
    with pytest.raises(ValueError, match="layers"):
        parse_llama_trace(payload)
    with pytest.raises(ValueError, match="unique"):
        replay_expert_cache(((1,) * 8,), slots=8)


def test_h4_summary_unions_repeated_misses_within_each_layer_block():
    replay = {
        "tokens": 8,
        "layers": {
            str(layer): {
                "per_token": [
                    {"misses": [1, 2]},
                    {"misses": [2, 3]},
                    {"misses": [3]},
                    {"misses": []},
                    {"misses": [4]},
                    {"misses": [4]},
                    {"misses": []},
                    {"misses": [5]},
                ]
            }
            for layer in range(40)
        },
    }
    summary = summarize_h4_miss_groups(replay, warmup_tokens=4)
    assert summary["blocks"][0]["sum_unique_miss_groups"] == 3 * 40
    assert summary["warm"]["mean_unique_miss_groups_per_layer_h4"] == 2.0
    assert summary["warm"]["mean_unique_miss_groups_all_layers_h4"] == 80.0
