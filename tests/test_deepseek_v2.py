from moe_lab.deepseek_v2 import expected_moe_layout, layer_prefix


def test_expected_v2_lite_moe_layout() -> None:
    layout = expected_moe_layout(
        {
            "hidden_size": 2048,
            "moe_intermediate_size": 1408,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 27,
            "first_k_dense_replace": 1,
        }
    )
    assert layout["moe_layers"] == 26
    assert layout["per_expert_parameters"] == 8_650_752
    assert layout["active_routed_parameters_per_token_per_layer"] == 51_904_512
    assert layout["expected_expert_shapes"]["down_proj.weight"] == [2048, 1408]


def test_layer_prefix_rejects_negative_layer() -> None:
    try:
        layer_prefix(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative layer must fail")
