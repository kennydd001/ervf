from pathlib import Path

import torch

from moe_lab.qwen_gptq_bank import (
    batched_official_gptq_projection,
    codes_from_quantized,
    fastgrid_pure_gptq_projection,
    nosync_pure_gptq_projection,
    official_pure_gptq_projection,
    pack_2bit_codes,
    unpack_2bit_codes,
)


GSQ_ROOT = Path(__file__).resolve().parents[2] / "third_party/GSQ"


def test_pack_roundtrip_all_codes():
    codes = torch.tensor(
        [[[-2, -1, 0, 1, 1, 0, -1, -2], [1, 1, -2, 0, -1, 0, 1, -2]]],
        dtype=torch.int8,
    )
    assert torch.equal(unpack_2bit_codes(pack_2bit_codes(codes)), codes)


def test_batched_matches_official_codes_and_bf16_scales_on_small_cpu_case():
    torch.manual_seed(260811)
    weights = (torch.randn(2, 8, 128) / 20).to(torch.bfloat16)
    calibration = torch.randn(2, 128, 128).to(torch.bfloat16)
    batched = batched_official_gptq_projection(weights, calibration, GSQ_ROOT)
    batched_codes = codes_from_quantized(batched.weight, batched.scales)
    for expert in range(2):
        official = official_pure_gptq_projection(weights[expert], calibration[expert], GSQ_ROOT)
        official_codes = codes_from_quantized(
            official.weight.unsqueeze(0), official.scales.to(torch.bfloat16).unsqueeze(0)
        )[0]
        assert torch.equal(batched_codes[expert], official_codes)
        assert torch.equal(
            batched.scales[expert].view(torch.uint16),
            official.scales.to(torch.bfloat16).view(torch.uint16),
        )
        fast = fastgrid_pure_gptq_projection(weights[expert], calibration[expert], GSQ_ROOT)
        fast_codes = codes_from_quantized(fast.weight.unsqueeze(0), fast.scales.unsqueeze(0))[0]
        assert torch.equal(fast_codes, official_codes)
        assert torch.equal(fast.scales.view(torch.uint16), official.scales.to(torch.bfloat16).view(torch.uint16))
        nosync = nosync_pure_gptq_projection(weights[expert], calibration[expert], GSQ_ROOT)
        nosync_codes = codes_from_quantized(nosync.weight.unsqueeze(0), nosync.scales.unsqueeze(0))[0]
        assert torch.equal(nosync_codes, official_codes)
        assert torch.equal(nosync.scales.view(torch.uint16), official.scales.to(torch.bfloat16).view(torch.uint16))
