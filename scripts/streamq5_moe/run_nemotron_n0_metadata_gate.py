from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from moe_lab.reporting import ROOT

REPORTS = ROOT / "reports/streamq5_moe"
OUTPUT = REPORTS / "nemotron_n0_metadata_gate.json"
REPORT = REPORTS / "NEMOTRON_N0_METADATA_GATE_REPORT_2026-08-12.md"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
REVISION = "main"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
FILES = ("config.json", "model.safetensors.index.json", "hf_quant_config.json")


def fetch(name: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(BASE + name, headers={"User-Agent": "STREAMQ5-metadata-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def nested(value: dict, *keys: str):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite NEM0 result")
    fetched = {}
    parsed = {}
    for name in FILES:
        try:
            payload, headers = fetch(name)
            fetched[name] = {
                "url": BASE + name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "etag": headers.get("etag"),
                "x_repo_commit": headers.get("x-repo-commit"),
            }
            parsed[name] = json.loads(payload)
        except Exception as exc:
            fetched[name] = {"url": BASE + name, "error": f"{type(exc).__name__}: {exc}"}

    config = parsed.get("config.json", {})
    index = parsed.get("model.safetensors.index.json", {})
    quant = parsed.get("hf_quant_config.json", {})
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    shards = sorted(set(weight_map.values()))
    metadata = index.get("metadata", {}) if isinstance(index, dict) else {}
    total_size = metadata.get("total_size")
    fields = {
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "num_experts": config.get("n_routed_experts", config.get("num_experts")),
        "top_k": config.get("num_experts_per_tok", config.get("num_experts_per_token")),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": config.get("shared_expert_intermediate_size"),
        "hybrid_override_pattern": config.get("hybrid_override_pattern"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "quantization_config": config.get("quantization_config"),
    }
    result = {
        "kind": "nemotron_n0_metadata_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision_requested": REVISION,
        "files": fetched,
        "identity": {
            "commit_values": sorted(set(row.get("x_repo_commit") for row in fetched.values() if row.get("x_repo_commit"))),
            "config": fields,
            "weight_map_tensors": len(weight_map),
            "shards": shards,
            "shard_count": len(shards),
            "index_total_size_bytes": total_size,
            "index_total_size_gib": total_size / 2**30 if isinstance(total_size, int) else None,
            "hf_quant_config": quant,
        },
        "gates": {
            "config_and_index_fetched": "config.json" in parsed and "model.safetensors.index.json" in parsed,
            "single_commit": len(set(row.get("x_repo_commit") for row in fetched.values() if row.get("x_repo_commit"))) == 1,
            "five_shards": len(shards) == 5,
            "model_payload_lt_25gib": isinstance(total_size, int) and total_size < 25 * 2**30,
            "lightning_alias_weight_identity_proven": False,
        },
        "status": "public_checkpoint_metadata_pass_alias_unproven",
        "claim_boundary": "Metadata/index-only network gate; no shard payload downloaded, tensor decoded, API output compared, kernel run, model quality or performance measured.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Nemotron N0 metadata gate\n\n"
        f"Public checkpoint: `{MODEL}`. Commit(s): `{result['identity']['commit_values']}`. "
        f"Index: {len(weight_map):,} tensors, {len(shards)} shards, {result['identity']['index_total_size_gib']} GiB.\n\n"
        "The NVIDIA NIM name `nemotron-3.5-nano-30b-a3b` exists, but public metadata does not prove that its optimized payload is byte-identical to this Hugging Face checkpoint. That alias gate remains open. No model shard was downloaded.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
