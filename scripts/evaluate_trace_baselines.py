from __future__ import annotations

import argparse
from pathlib import Path

from moe_lab.deepseek_v2 import expected_moe_layout, load_json
from moe_lab.metrics import regression_metrics
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import load_trace, trace_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--name", default="trace_baselines.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trace = load_trace(args.trace)
    config = load_json(ROOT / "models" / "deepseek-v2-lite" / "config.json")
    layout = expected_moe_layout(config)
    reference_bytes = (
        layout["active_routed_parameters_per_token_per_layer"] * 2
    )
    one_expert_bytes = layout["per_expert_parameters"] * 2
    results = {}
    for name, prediction in trace_baselines(trace).items():
        results[name] = regression_metrics(prediction, trace.routed_output)
        results[name]["estimated_bf16_expert_bytes_per_token"] = (
            0 if name == "zero" else one_expert_bytes
        )
        results[name]["compression_ratio_vs_topk_bf16"] = (
            float("inf") if name == "zero" else reference_bytes / one_expert_bytes
        )
    results["exact_topk"] = {
        **regression_metrics(trace.routed_output, trace.routed_output),
        "estimated_bf16_expert_bytes_per_token": reference_bytes,
        "compression_ratio_vs_topk_bf16": 1.0,
    }
    payload = {
        "trace": str(args.trace.resolve()),
        "validation": trace.validate(),
        "results": results,
        "caveat": (
            "Byte counts model compulsory BF16 expert-weight reads only; they are not "
            "measured runtime traffic and exclude cache, router, shared expert and attention."
        ),
    }
    path = write_json(args.name, envelope("trace_baselines", payload))
    print(path)
    for name, metrics in results.items():
        print(f"{name}: nrmse={metrics['nrmse']:.6f}, cosine={metrics['cosine']:.6f}")

