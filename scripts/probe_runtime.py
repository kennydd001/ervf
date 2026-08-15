from __future__ import annotations

import traceback
from importlib import metadata

from moe_lab.reporting import ROOT, envelope, write_json


def attempt(name: str, function):
    try:
        value = function()
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    return {"name": name, "ok": True, "value": value}


def torch_probe():
    import torch
    import torch._dynamo

    return {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dynamo_module": torch._dynamo.__file__,
    }


def transformers_probe():
    from transformers import AutoConfig, AutoTokenizer

    model_dir = ROOT / "models" / "deepseek-v2-lite"
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    return {
        "config_class": type(config).__name__,
        "hidden_size": config.hidden_size,
        "attention_implementation": config._attn_implementation,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "sample_ids": tokenizer.encode("Runtime probe."),
    }


if __name__ == "__main__":
    payload = {
        "installed_versions": {
            package: metadata.version(package)
            for package in ("torch", "transformers", "huggingface-hub", "tokenizers")
        },
        "probes": [
            attempt("torch_and_dynamo", torch_probe),
            attempt("deepseek_config_and_tokenizer", transformers_probe),
        ],
    }
    path = write_json("runtime_probe.json", envelope("runtime_probe", payload))
    print(path)
    for probe in payload["probes"]:
        print(f"{probe['name']}: {'ok' if probe['ok'] else probe['error_type']}")
    if not all(probe["ok"] for probe in payload["probes"]):
        raise SystemExit(1)

