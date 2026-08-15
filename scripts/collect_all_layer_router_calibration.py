from __future__ import annotations

import gc

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.moe_layer import load_token_embeddings
from moe_lab.partial_forward import load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


BLOCK_SIZE = 128
BLOCKS = 16


def train_blocks(model_dir):
    parquet = (
        ROOT
        / "data"
        / "corpora"
        / "wikitext"
        / "wikitext-2-raw-v1"
        / "train-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = tokenizer.encode(joined).ids[: BLOCK_SIZE * BLOCKS]
    return torch.tensor(ids, dtype=torch.long).view(BLOCKS, BLOCK_SIZE)


@torch.inference_mode()
def forward_and_capture(layer, hidden_states):
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured = []

    def hook(_module, _inputs, output):
        captured.append((output[0].detach().cpu(), output[1].detach().cpu()))

    handle = layer.mlp.gate.register_forward_hook(hook)
    try:
        output = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
    finally:
        handle.remove()
    return output, captured[0]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("router calibration requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    input_ids = train_blocks(model_dir)
    hidden = load_token_embeddings(model_dir, input_ids, device)
    batch, sequence, _ = hidden.shape
    position_ids = torch.arange(sequence, device=device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(None, (batch, sequence), hidden, 0)
    layer_zero, _ = load_decoder_layer(model_dir, 0, device)
    hidden = layer_zero(
        hidden,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
    )[0]
    del layer_zero
    gc.collect()
    torch.cuda.empty_cache()

    layers = []
    for layer_idx in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        hidden, (ids, weights) = forward_and_capture(layer, hidden)
        mass = torch.zeros(64, dtype=torch.float64)
        counts = torch.zeros(64, dtype=torch.int64)
        mass.scatter_add_(0, ids.long().reshape(-1), weights.double().reshape(-1))
        counts.scatter_add_(
            0, ids.long().reshape(-1), torch.ones(ids.numel(), dtype=torch.int64)
        )
        ordering = torch.argsort(mass, descending=True)
        hot = ordering[:32]
        layers.append(
            {
                "layer": layer_idx,
                "router_mass": mass.tolist(),
                "router_counts": counts.tolist(),
                "hot_expert_ids": sorted(hot.tolist()),
                "hot_router_mass_fraction": float(mass[hot].sum() / mass.sum()),
            }
        )
        print(
            f"layer={layer_idx:02d} hot_mass={layers[-1]['hot_router_mass_fraction']:.6f}",
            flush=True,
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "split": "train",
        "blocks": BLOCKS,
        "block_size": BLOCK_SIZE,
        "tokens": BLOCKS * BLOCK_SIZE,
        "selection": "top 32 experts per layer by sum of unrenormalized router weights",
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "layers": layers,
    }
    path = write_json(
        "router_calibration_all_layers.json", envelope("router_calibration", report)
    )
    print(path)
