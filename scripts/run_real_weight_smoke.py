from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from moe_lab.moe_layer import load_moe_layer, load_token_embeddings
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import save_trace


SMOKE_TEXT = """
Mixture-of-experts models route every token to a small subset of experts.
This smoke test uses real DeepSeek weights but token embeddings instead of true
layer-one activations. It validates checkpoint loading, routing, expert
execution, trace storage and baseline metrics; it is not scientific evidence
for activation-space compression.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    token_ids = tokenizer.encode(SMOKE_TEXT).ids[: args.max_tokens]
    if not token_ids:
        raise RuntimeError("smoke text produced no tokens")
    ids = torch.tensor(token_ids, dtype=torch.long)
    hidden_states = load_token_embeddings(model_dir, ids, device)
    moe = load_moe_layer(model_dir, args.layer, device)
    trace = moe.trace(hidden_states)
    trace_path = ROOT / "data" / "traces" / f"real_weight_smoke_layer_{args.layer}.safetensors"
    validation = save_trace(
        trace,
        trace_path,
        {
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "layer": args.layer,
            "activation_source": "token_embeddings_not_true_layer_activations",
            "device": str(device),
        },
    )
    report = {
        "status": "complete",
        "trace": str(trace_path.resolve()),
        "validation": validation,
        "activation_source": "token_embeddings_not_true_layer_activations",
        "scientific_evidence": False,
    }
    path = write_json(
        f"real_weight_smoke_layer_{args.layer}.json",
        envelope("real_weight_smoke", report),
    )
    print(path)
    print(validation)

