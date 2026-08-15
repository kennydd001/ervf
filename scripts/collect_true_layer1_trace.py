from __future__ import annotations

import argparse
import gc

import torch
from tokenizers import Tokenizer

from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import layer_moe_input, load_decoder_layer, run_layer_zero
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import save_trace


TRACE_TEXT = """
Mixture-of-experts models route each token to a small subset of feed-forward
networks. A valid compression experiment must preserve both the selected expert
outputs and the downstream routing decisions during autoregressive generation.
This calibration sentence exercises the exact embedding, dense layer zero and
attention path that produce the input to DeepSeek V2 Lite's first MoE layer.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    token_ids = tokenizer.encode(TRACE_TEXT).ids[: args.max_tokens]
    if not token_ids:
        raise RuntimeError("trace text produced no tokens")
    ids = torch.tensor(token_ids, dtype=torch.long)
    embeddings = load_token_embeddings(model_dir, ids, device).unsqueeze(0)

    layer_zero, _ = load_decoder_layer(model_dir, 0, device)
    hidden_after_zero = run_layer_zero(layer_zero, embeddings)
    del layer_zero, embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    layer_one, _ = load_decoder_layer(model_dir, 1, device)
    true_moe_input = layer_moe_input(layer_one, hidden_after_zero)
    moe = loaded_moe_from_official_module(layer_one.mlp, layer=1)
    trace = moe.trace(true_moe_input)

    trace_path = ROOT / "data" / "traces" / "true_activation_smoke_layer_1.safetensors"
    validation = save_trace(
        trace,
        trace_path,
        {
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "layer": 1,
            "activation_source": "exact_embedding_layer0_and_layer1_attention_local_text",
            "device": str(device),
            "token_count": len(token_ids),
        },
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    report = {
        "status": "complete",
        "trace": str(trace_path.resolve()),
        "validation": validation,
        "activation_source": "exact_embedding_layer0_and_layer1_attention_local_text",
        "scientific_evidence": False,
        "reason_not_scientific": "pipeline smoke on one locally authored text",
        "peak_cuda_allocated_bytes": peak_bytes,
    }
    path = write_json("true_activation_smoke_layer_1.json", envelope("true_activation_smoke", report))
    print(path)
    print(validation)
    print({"peak_cuda_allocated_bytes": peak_bytes})
