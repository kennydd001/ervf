from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from moe_lab.reporting import ROOT


MODEL = ROOT / "models/qwen3-30b-a3b-base"
CORPUS = ROOT / "data/corpora/hera_moe_p0"
PREREG = ROOT / "reports/streamq5_moe/P1C_CORRECTED_ROUTE_CACHE_PREREGISTRATION.md"
P0_VERIFY = ROOT / "reports/streamq5_moe/p0c_model_quality_verification.json"
ARTIFACT = ROOT / "reports/runs/streamq5_moe/p1c_fresh_route_input_ids.safetensors"
OUTPUT = ROOT / "reports/streamq5_moe/p1c_route_input_lock.json"
PRIOR_INPUTS = (
    ROOT / "reports/runs/coretail_moe/p2_heldout_input_ids.safetensors",
    ROOT / "reports/runs/streamq4_moe/p0_fresh_input_ids.safetensors",
    ROOT / "reports/runs/streamq5_moe/p0_fresh_input_ids.safetensors",
    ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors",
    ROOT / "reports/runs/streamq5_moe/p1a_fresh_route_input_ids.safetensors",
)
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LANGUAGES = ("arb_Arab", "zho_Hans", "hin_Deva", "rus_Cyrl", "spa_Latn", "swh_Latn", "jpn_Jpan", "nld_Latn")
TOKENS = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def take(values: list[int], offset: int, count: int, label: str) -> list[int]:
    result = values[offset : offset + count]
    if len(result) != count:
        raise RuntimeError(f"{label}: requested {count} at {offset}, got {len(result)} of {len(values)}")
    return result


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite P1C route input lock")
    p0 = json.loads(P0_VERIFY.read_text(encoding="utf-8"))
    if p0.get("status") != "p0c_quality_verification_pass":
        raise RuntimeError("independent STREAMQ5 P0C quality pass required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    tensors = {}
    manifest = {}

    wiki = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    rows = pq.read_table(wiki, columns=["text"])["text"].to_pylist()
    ids = encode(tokenizer, "\n\n".join(x for x in rows if x and x.strip()))
    offset = 320_000
    tensors["general"] = torch.tensor(take(ids, offset, TOKENS, "general"), dtype=torch.int64).reshape(1, TOKENS)
    manifest["general"] = {"path": str(wiki.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(wiki), "offset": offset, "tokens": TOKENS, "total_tokens": len(ids)}

    code = []
    code_manifest = []
    for language in ("python", "java"):
        path = CORPUS / f"code/{language}/train-00000-of-00001.parquet"
        rows = pq.read_table(path, columns=["input"])["input"].to_pylist()
        ids = encode(tokenizer, "\n\n".join(x for x in rows if x))
        offset = 240_000
        code.extend(take(ids, offset, TOKENS // 2, f"code_{language}"))
        code_manifest.append({"language": language, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "offset": offset, "tokens": TOKENS // 2, "total_tokens": len(ids)})
    tensors["code"] = torch.tensor(code, dtype=torch.int64).reshape(1, TOKENS)
    manifest["code"] = code_manifest

    math_path = CORPUS / "math/main/train-00000-of-00001.parquet"
    table = pq.read_table(math_path, columns=["question", "answer"])
    ids = encode(tokenizer, "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in zip(table["question"].to_pylist(), table["answer"].to_pylist())))
    offset = 650_000
    tensors["math"] = torch.tensor(take(ids, offset, TOKENS, "math"), dtype=torch.int64).reshape(1, TOKENS)
    manifest["math"] = {"path": str(math_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(math_path), "offset": offset, "tokens": TOKENS, "total_tokens": len(ids)}

    instruction_path = CORPUS / "instruction/databricks-dolly-15k.jsonl"
    text = []
    with instruction_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text.append(f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}")
    ids = encode(tokenizer, "\n\n".join(text))
    offset = 950_000
    tensors["instruction"] = torch.tensor(take(ids, offset, TOKENS, "instruction"), dtype=torch.int64).reshape(1, TOKENS)
    manifest["instruction"] = {"path": str(instruction_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(instruction_path), "offset": offset, "tokens": TOKENS, "total_tokens": len(ids)}

    multilingual_path = CORPUS / "multilingual/dev.parquet"
    table = pq.read_table(multilingual_path, columns=list(LANGUAGES))
    multi = []
    multi_manifest = []
    for language in LANGUAGES:
        ids = encode(tokenizer, "\n\n".join(x for x in table[language].to_pylist() if x))
        offset = 26_000
        multi.extend(take(ids, offset, TOKENS // len(LANGUAGES), f"multilingual_{language}"))
        multi_manifest.append({"language": language, "offset": offset, "tokens": TOKENS // len(LANGUAGES), "total_tokens": len(ids)})
    tensors["multilingual"] = torch.tensor(multi, dtype=torch.int64).reshape(1, TOKENS)
    manifest["multilingual"] = {"path": str(multilingual_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(multilingual_path), "languages": multi_manifest}

    if set(tensors) != set(DOMAINS) or any(tuple(value.shape) != (1, TOKENS) for value in tensors.values()):
        raise RuntimeError("P1C route input contract failed")
    prior = [load_file(path) for path in PRIOR_INPUTS]
    disjoint = True
    for domain in DOMAINS:
        chunks = tensors[domain].reshape(-1, 128)
        for old in prior:
            old_rows = [
                value
                for name, tensor in old.items()
                if name == domain or name.endswith(f"_{domain}")
                for value in tensor.reshape(-1, 128)
            ]
            disjoint &= not any((left == right).all().item() for left in chunks for right in old_rows)
    if not disjoint:
        raise RuntimeError("P1C route context overlaps a prior decision context")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={"kind": "streamq5_moe_p1c_fresh_route_inputs", "routes_opened": "false"})
    payload = {
        "kind": "streamq5_moe_p1c_route_input_lock", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "p0_verification_sha256": sha256(P0_VERIFY),
        "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": sha256(ARTIFACT),
        "domains": list(DOMAINS), "context_tokens": TOKENS,
        "partitions": {"calibration": [0, 512], "validation": [512, 768], "test": [768, 1024]},
        "input_ids_sha256": {name: tensor_sha(value) for name, value in tensors.items()},
        "source_manifest": manifest, "prior_input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in PRIOR_INPUTS},
        "exact_128_context_disjoint_from_prior_decisions": disjoint,
        "routes_opened": False,
        "cache": {"total_slots": 1910, "static_slots_per_layer": 32, "dynamic_slots_layers_0_37": 8, "dynamic_slots_layers_38_47": 7, "expert_record_bytes": 3035136},
        "gates": {"mean_h2d_ms_max": 25.0, "p95_h2d_ms_max": 35.0, "static_preload_ms_max": 250.0, "bandwidth_gb_s": 26.158915272090432},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "locked", "artifact": payload["artifact"], "sha256": payload["artifact_sha256"], "disjoint": disjoint, "manifest": manifest}, indent=2))
