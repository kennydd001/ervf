from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import layer_moe_input, load_decoder_layer, run_layer_zero
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import save_trace


DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--train-tokens", type=int, default=16_384)
    parser.add_argument("--eval-tokens", type=int, default=4_096)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def token_blocks(
    parquet_path: Path, tokenizer: Tokenizer, block_size: int, token_limit: int
) -> torch.Tensor:
    texts = pq.read_table(parquet_path, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    ids = tokenizer.encode(joined).ids
    usable = min(len(ids), token_limit)
    usable -= usable % block_size
    if usable == 0:
        raise RuntimeError(f"not enough tokens for one block in {parquet_path}")
    return torch.tensor(ids[:usable], dtype=torch.long).view(-1, block_size)


if __name__ == "__main__":
    args = parse_args()
    if (
        args.block_size <= 0
        or args.train_tokens < args.block_size
        or args.eval_tokens < args.block_size
    ):
        raise ValueError("each split must contain at least one positive block")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    corpus_dir = ROOT / "data" / "corpora" / "wikitext" / "wikitext-2-raw-v1"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    blocks = {
        split: token_blocks(
            corpus_dir / f"{split}-00000-of-00001.parquet",
            tokenizer,
            args.block_size,
            args.train_tokens if split == "train" else args.eval_tokens,
        )
        for split in ("train", "validation", "test")
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    flat_ids = torch.cat([value.reshape(-1) for value in blocks.values()])
    all_embeddings = load_token_embeddings(model_dir, flat_ids, device)
    embeddings: dict[str, torch.Tensor] = {}
    offset = 0
    for split, split_blocks in blocks.items():
        count = split_blocks.numel()
        embeddings[split] = all_embeddings[offset : offset + count].view(
            split_blocks.shape[0], split_blocks.shape[1], -1
        )
        offset += count
    del all_embeddings

    layer_zero, _ = load_decoder_layer(model_dir, 0, device)
    hidden_after_zero = {
        split: run_layer_zero(layer_zero, value) for split, value in embeddings.items()
    }
    del layer_zero, embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    layer_one, _ = load_decoder_layer(model_dir, 1, device)
    moe = loaded_moe_from_official_module(layer_one.mlp, layer=1)
    split_reports = {}
    for split, hidden_states in hidden_after_zero.items():
        exact_input = layer_moe_input(layer_one, hidden_states)
        trace = moe.trace(exact_input)
        trace_path = ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors"
        validation = save_trace(
            trace,
            trace_path,
            {
                "model_revision": MODEL_REVISION,
                "dataset_id": "Salesforce/wikitext",
                "dataset_revision": DATASET_REVISION,
                "dataset_config": "wikitext-2-raw-v1",
                "split": split,
                "layer": 1,
                "block_size": args.block_size,
                "token_count": int(blocks[split].numel()),
                "activation_source": "exact_embedding_layer0_and_layer1_attention",
            },
        )
        split_reports[split] = {
            "trace": str(trace_path.resolve()),
            "validation": validation,
        }

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    report = {
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "block_size": args.block_size,
        "requested_train_tokens": args.train_tokens,
        "requested_eval_tokens": args.eval_tokens,
        "device": str(device),
        "peak_cuda_allocated_bytes": peak_bytes,
        "splits": split_reports,
    }
    path = write_json("wikitext_layer1_traces.json", envelope("real_activation_traces", report))
    print(path)
    print({"peak_cuda_allocated_bytes": peak_bytes})
    for split, result in split_reports.items():
        print(split, result["validation"])
