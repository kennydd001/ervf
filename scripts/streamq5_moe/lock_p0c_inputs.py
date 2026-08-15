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
PREREG = ROOT / "reports/streamq5_moe/P0C_PHYSICAL_SEMANTICS_PREREGISTRATION.md"
OLD_LOCK = ROOT / "reports/coretail_moe/p2_input_lock.json"
OLD_INPUT = ROOT / "reports/runs/coretail_moe/p2_heldout_input_ids.safetensors"
Q4_LOCK = ROOT / "reports/streamq4_moe/p0_input_lock.json"
Q4_INPUT = ROOT / "reports/runs/streamq4_moe/p0_fresh_input_ids.safetensors"
Q5_LOCK = ROOT / "reports/streamq5_moe/p0_input_lock.json"
Q5_INPUT = ROOT / "reports/runs/streamq5_moe/p0_fresh_input_ids.safetensors"
ARTIFACT = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
OUTPUT = ROOT / "reports/streamq5_moe/p0c_input_lock.json"
CONTEXT = 128
PER_SPLIT = 256
LANGUAGES = ("arb_Arab", "zho_Hans", "hin_Deva", "rus_Cyrl", "spa_Latn", "swh_Latn", "jpn_Jpan", "nld_Latn")
DOMAINS = ("general", "code", "math", "multilingual", "instruction")


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
        raise RuntimeError(f"{label}: requested {count} tokens at {offset}, got {len(result)} of {len(values)}")
    return result


def contexts(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64).reshape(-1, CONTEXT)


def no_exact_context_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    return not any((a == b).all().item() for a in left for b in right)


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite STREAMQ5 P0C input lock")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {}

    wiki = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    wiki_rows = pq.read_table(wiki, columns=["text"])["text"].to_pylist()
    wiki_ids = encode(tokenizer, "\n\n".join(x for x in wiki_rows if x and x.strip()))
    general_offset = 264_192
    general = take(wiki_ids, general_offset, 2 * PER_SPLIT, "general")
    tensors["validation_general"] = contexts(general[:PER_SPLIT])
    tensors["test_general"] = contexts(general[PER_SPLIT:])
    manifest["general"] = {
        "path": str(wiki.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(wiki),
        "total_tokens": len(wiki_ids), "validation_offset": general_offset,
        "test_offset": general_offset + PER_SPLIT, "tokens_per_split": PER_SPLIT,
    }

    code_parts = {"validation": [], "test": []}
    code_manifest = {}
    for language in ("python", "java"):
        path = CORPUS / f"code/{language}/train-00000-of-00001.parquet"
        rows = pq.read_table(path, columns=["input"])["input"].to_pylist()
        ids = encode(tokenizer, "\n\n".join(x for x in rows if x))
        offset = 198_144
        chosen = take(ids, offset, PER_SPLIT, f"code_{language}")
        code_parts["validation"].extend(chosen[: PER_SPLIT // 2])
        code_parts["test"].extend(chosen[PER_SPLIT // 2 :])
        code_manifest[language] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path),
            "total_tokens": len(ids), "validation_offset": offset,
            "test_offset": offset + PER_SPLIT // 2, "tokens_per_split": PER_SPLIT // 2,
        }
    for split in ("validation", "test"):
        tensors[f"{split}_code"] = contexts(code_parts[split])
    manifest["code"] = code_manifest

    math_path = CORPUS / "math/main/train-00000-of-00001.parquet"
    math_table = pq.read_table(math_path, columns=["question", "answer"])
    math_text = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in zip(math_table["question"].to_pylist(), math_table["answer"].to_pylist()))
    math_ids = encode(tokenizer, math_text)
    math_offset = 526_336
    math = take(math_ids, math_offset, 2 * PER_SPLIT, "math")
    tensors["validation_math"] = contexts(math[:PER_SPLIT])
    tensors["test_math"] = contexts(math[PER_SPLIT:])
    manifest["math"] = {
        "path": str(math_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(math_path),
        "total_tokens": len(math_ids), "validation_offset": math_offset,
        "test_offset": math_offset + PER_SPLIT, "tokens_per_split": PER_SPLIT,
    }

    instruction_path = CORPUS / "instruction/databricks-dolly-15k.jsonl"
    instruction_rows = []
    with instruction_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}")
    instruction_ids = encode(tokenizer, "\n\n".join(instruction_rows))
    instruction_offset = 788_480
    instruction = take(instruction_ids, instruction_offset, 2 * PER_SPLIT, "instruction")
    tensors["validation_instruction"] = contexts(instruction[:PER_SPLIT])
    tensors["test_instruction"] = contexts(instruction[PER_SPLIT:])
    manifest["instruction"] = {
        "path": str(instruction_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(instruction_path),
        "total_tokens": len(instruction_ids), "validation_offset": instruction_offset,
        "test_offset": instruction_offset + PER_SPLIT, "tokens_per_split": PER_SPLIT,
    }

    multilingual_path = CORPUS / "multilingual/dev.parquet"
    multilingual_table = pq.read_table(multilingual_path, columns=list(LANGUAGES))
    multi_parts = {"validation": [], "test": []}
    multi_manifest = {}
    for language in LANGUAGES:
        ids = encode(tokenizer, "\n\n".join(x for x in multilingual_table[language].to_pylist() if x))
        offset = 25_088
        per_language_split = PER_SPLIT // len(LANGUAGES)
        chosen = take(ids, offset, 2 * per_language_split, f"multilingual_{language}")
        multi_parts["validation"].extend(chosen[:per_language_split])
        multi_parts["test"].extend(chosen[per_language_split:])
        multi_manifest[language] = {
            "total_tokens": len(ids), "validation_offset": offset,
            "test_offset": offset + per_language_split, "tokens_per_split": per_language_split,
        }
    for split in ("validation", "test"):
        tensors[f"{split}_multilingual"] = contexts(multi_parts[split])
    manifest["multilingual"] = {
        "path": str(multilingual_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(multilingual_path), "languages": multi_manifest,
    }

    expected = {f"{split}_{domain}" for split in ("validation", "test") for domain in DOMAINS}
    if set(tensors) != expected or any(tuple(value.shape) != (2, CONTEXT) for value in tensors.values()):
        raise RuntimeError("STREAMQ5 tensor contract failed")
    old = load_file(OLD_INPUT)
    q4 = load_file(Q4_INPUT)
    q5 = load_file(Q5_INPUT)
    all_disjoint = True
    for domain in DOMAINS:
        all_disjoint &= no_exact_context_overlap(tensors[f"validation_{domain}"], tensors[f"test_{domain}"])
        for new_split in ("validation", "test"):
            for old_split in ("validation", "test"):
                all_disjoint &= no_exact_context_overlap(tensors[f"{new_split}_{domain}"], old[f"{old_split}_{domain}"])
                all_disjoint &= no_exact_context_overlap(tensors[f"{new_split}_{domain}"], q4[f"{old_split}_{domain}"])
                all_disjoint &= no_exact_context_overlap(tensors[f"{new_split}_{domain}"], q5[f"{old_split}_{domain}"])
    if not all_disjoint:
        raise RuntimeError("P0C split overlaps an earlier exact decision context")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={"kind": "streamq5_moe_p0c_fresh_inputs", "outputs_opened": "false"})
    payload = {
        "kind": "streamq5_moe_p0c_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "previous_coretail_lock_sha256": sha256(OLD_LOCK),
        "previous_coretail_input_sha256": sha256(OLD_INPUT),
        "previous_streamq4_lock_sha256": sha256(Q4_LOCK),
        "previous_streamq4_input_sha256": sha256(Q4_INPUT),
        "previous_streamq5_lock_sha256": sha256(Q5_LOCK),
        "previous_streamq5_input_sha256": sha256(Q5_INPUT),
        "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT),
        "splits": ["validation", "test"], "domains": list(DOMAINS),
        "contexts_per_domain": 2, "context_tokens": CONTEXT, "tokens_per_domain": PER_SPLIT,
        "input_ids_sha256": {name: tensor_sha(value) for name, value in tensors.items()},
        "source_manifest": manifest, "outputs_opened": False,
        "exact_context_disjoint_from_prior_decision_sets": all_disjoint,
        "physical_semantics": "codes selected with FP32 maxabs scale; scale stored BF16; dequant uses float(BF16 scale) then BF16 output",
        "variants": ["bf16_teacher", "q5_experts_bf16_trunk", "bf16_experts_int8_trunk", "q5_experts_int8_trunk", "q5_experts_int4_trunk"],
        "quantization": {
            "group_size": 128, "q5_codes": [-15, 15], "int8_codes": [-127, 127],
            "int4_codes": [-7, 7], "rounding": "torch_round_nearest_even", "dequant_dtype": "bfloat16",
        },
        "primary_gate": {"variant": "q5_experts_int8_trunk", "validation_progression_relative_ce_max": 0.025, "validation_relative_ce_max": 0.02, "test_relative_ce_max": 0.02},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "locked", "artifact": payload["artifact"], "sha256": payload["artifact_sha256"], "fresh_disjoint": all_disjoint, "manifest": manifest}, indent=2))
