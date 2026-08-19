from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import numpy as np

from common import REPO, require_model_dir, sha256_file

RESULTS = REPO / "pro_research" / "results" / "s100_lightning15"
PROMPTS = REPO / "pro_research" / "S100_PHASE3_PROMPTS.json"
OLD_TRACE = (
    REPO / "pro_research" / "results"
    / "S100_PHASE3_V18_TRACE_FULL.npz"
)
OLD_TRACE_META = OLD_TRACE.with_suffix(".json")
TOP_K = 64

SPLITS = {
    "calibration": ("_01", 64),
    "validation": ("_02", 128),
    "heldout": (("_03", "_04"), 256),
}

STRICT = {
    "top1": 0.970,
    "top5": 0.999,
    "mean_ce": 0.025,
    "mean_kl": 0.015,
    "p95_kl": 0.060,
    "domain_top1": 0.90,
    "domain_ce": 0.080,
}

OFFICIAL = {
    "top1": 0.95,
    "top5": 0.995,
    "mean_ce": 0.05,
    "p95_ce": 0.25,
    "mean_kl": 0.02,
    "p95_kl": 0.08,
    "domain_top1": 0.90,
    "domain_ce": 0.10,
}

def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS

def identity() -> dict:
    model = require_model_dir()
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def get(name, default=None):
        if name in config:
            return config[name]
        text = config.get("text_config")
        if isinstance(text, dict) and name in text:
            return text[name]
        return default

    record = {
        "model_dir": str(model),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "max_position_embeddings": int(get("max_position_embeddings", -1)),
        "hidden_size": int(get("hidden_size", -1)),
        "num_hidden_layers": int(get("num_hidden_layers", -1)),
        "n_routed_experts": int(get("n_routed_experts", -1)),
        "num_experts_per_tok": int(get("num_experts_per_tok", -1)),
        "moe_intermediate_size": int(get("moe_intermediate_size", -1)),
        "vocab_size": int(get("vocab_size", -1)),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "shard_count": len(json.loads(
            index_path.read_text(encoding="utf-8")
        ).get("weight_map", {})),
    }
    expected = {
        "max_position_embeddings": 1_048_576,
        "hidden_size": 2688,
        "num_hidden_layers": 52,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "vocab_size": 131072,
    }
    mismatches = {
        key: {"expected": value, "actual": record[key]}
        for key, value in expected.items()
        if record[key] != value
    }
    record["expected"] = expected
    record["mismatches"] = mismatches
    record["LIGHTNING_IDENTITY_GREEN"] = not mismatches
    if mismatches:
        raise RuntimeError(
            "checkpoint is not the required Lightning identity: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return record

def prompt_rows(split: str) -> tuple[list[dict], int]:
    suffix, length = SPLITS[split]
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    suffixes = suffix if isinstance(suffix, tuple) else (suffix,)
    selected = [
        row for row in rows
        if row["id"].endswith(suffixes)
    ]
    expected = 20 if split == "heldout" else 10
    if len(selected) != expected:
        raise RuntimeError(
            f"{split}: expected {expected} prompts, got {len(selected)}"
        )
    return selected, length

def trace_paths(split: str):
    base = RESULTS / f"S100_LIGHTNING15_TRACE_{split.upper()}"
    return base.with_suffix(".npz"), base.with_suffix(".json")

def normalize_eager_moe(rt):
    original = rt._moe

    def safe(self, layer, out):
        result = original(layer, out)
        return (None, None) if result is None else result

    rt._moe = types.MethodType(safe, rt)
    return original

def reset_eager(rt):
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed_prompt(rt, ids):
    for token in ids:
        rt.step(int(token))

def logsumexp(cp, logits) -> float:
    maximum = cp.max(logits)
    return float(
        (maximum + cp.log(cp.exp(logits - maximum).sum())).item()
    )

def model_signature(identity_record: dict) -> str:
    stable = {
        key: identity_record[key]
        for key in (
            "config_sha256", "index_sha256",
            "max_position_embeddings", "hidden_size",
            "num_hidden_layers", "n_routed_experts",
            "num_experts_per_tok", "moe_intermediate_size",
            "vocab_size",
        )
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()
    ).hexdigest()

def candidate_name(mode: str, families: str) -> str:
    return f"{mode}_{families}".lower()

def parse_families(value: str) -> set[str]:
    value = value.lower()
    allowed = {"k", "v", "o"}
    result = set(value)
    if not result or not result <= allowed:
        raise ValueError(f"invalid families: {value}")
    return result
